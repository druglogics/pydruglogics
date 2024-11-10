import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import precision_recall_curve, auc
from pydruglogics.model.ModelPredictions import ModelPredictions
from pydruglogics.utils.PlotUtil import PlotUtil
from typing import List, Tuple, Optional, Any

def _normalize_synergy_scores(calibrated_synergy_scores: List[Tuple[str, float]],
                              prolif_synergy_scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Normalize synergy scores based on calibrated and proliferative synergy scores.
    :param calibrated_synergy_scores: List of tuples (perturbation, calibrated synergy score).
    :param prolif_synergy_scores: List of tuples (perturbation, proliferative synergy score).
    :return: List of tuples (perturbation, normalized synergy score).
    """
    normalized_synergy_scores = []
    for (perturbation, ss_score), (_, prolif_score) in zip(calibrated_synergy_scores, prolif_synergy_scores):
        normalized_synergy_score = math.exp(ss_score - prolif_score)
        normalized_synergy_scores.append((perturbation, normalized_synergy_score))

    return normalized_synergy_scores

def _bootstrap_resample(labels: np.ndarray, predictions: np.ndarray, boot_n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Resample data for bootstrapping.
    :param labels: Array of observed binary labels.
    :param predictions: Array of predicted scores.
    :param boot_n: Number of bootstrap resampling iterations.
    :return: List of tuples containing resampled labels and predictions.
    """
    resampled_model_preds = []
    for _ in range(boot_n):
        rnd = np.random.choice(len(labels), size=len(labels), replace=True)
        resampled_labels = labels[rnd]
        resampled_predictions = predictions[rnd]
        resampled_model_preds.append((resampled_labels, resampled_predictions))
    return resampled_model_preds

def _calculate_pr_with_ci(observed: np.ndarray, preds: np.ndarray, boot_n: int,
                         confidence_level: float, with_seeds: bool, seeds: int) -> Tuple[pd.DataFrame, float]:
    """
    Calculate Precision-Recall curve with confidence intervals.
    :param observed: Array of observed binary labels.
    :param preds: Array of predicted scores.
    :param boot_n: Number of bootstrap resampling iterations.
    :param confidence_level: Confidence level for calculating the confidence intervals (e.g., 0.9 for 90% CI).
    :param with_seeds: Whether to use a fixed seed for reproducibility of the bootstrap sampling.
    :param seeds: Seed value for random number generation to ensure reproducibility.
    :return: A tuple containing a DataFrame with 'recall', 'precision', 'low_precision', and 'high_precision'
    columns for confidence intervals the AUC-PR.
    """
    if with_seeds:
        np.random.seed(seeds)

    precision_orig, recall_orig, _ = precision_recall_curve(observed, preds)
    auc_pr = auc(recall_orig, precision_orig)
    pr_df = pd.DataFrame({'recall': recall_orig, 'precision': precision_orig})

    resampled_data = _bootstrap_resample(observed, preds, boot_n=boot_n)
    precision_matrix = []

    for resampled_observed, resampled_predicted in resampled_data:
        precision_boot, recall_boot, _ = precision_recall_curve(resampled_observed, resampled_predicted)
        const_interp_pr = interp1d(recall_boot, precision_boot, kind='previous', bounds_error=False,
                                   fill_value=(precision_boot[0], precision_boot[-1]))
        aligned_precisions = const_interp_pr(recall_orig)
        precision_matrix.append(aligned_precisions)

    precision_matrix = np.array(precision_matrix)

    alpha = 1 - confidence_level
    low_precision = np.percentile(precision_matrix, alpha / 2 * 100, axis=0)
    high_precision = np.percentile(precision_matrix, (1 - alpha / 2) * 100, axis=0)

    pr_df['low_precision'] = low_precision
    pr_df['high_precision'] = high_precision

    return pr_df, auc_pr

def sampling_with_ci(boolean_models: List, observed_synergy_scores: List[str], model_outputs: Any,
                     perturbations: Any, synergy_method: str = 'hsa', repeat_time: int = 10, sub_ratio: float = 0.8,
                     boot_n: int = 1000, confidence_level: float = 0.9, plot_discrete: bool = False,
                     with_seeds: bool = True, seeds: int = 42) -> None:
    """
    Perform sampling with confidence interval calculation and plot PR curve.
    :param boolean_models: List of BooleanModel instances.
    :param observed_synergy_scores: List of observed synergy scores.
    :param model_outputs: Model outputs for evaluation.
    :param perturbations: List of perturbations to apply to the models.
    :param synergy_method: Method to check for synergy ('hsa' or 'bliss').
    :param repeat_time: Number of times to repeat sampling.
    :param sub_ratio: Proportion of models to sample in each iteration.
    :param boot_n: Number of bootstrap resampling iterations for confidence intervals.
    :param confidence_level: Confidence level for confidence interval calculations.
    :param plot_discrete: Whether to plot discrete points on the PR curve.
    :param with_seeds: Whether to use a fixed seed for reproducibility.
    :param seeds: Seed value for random number generation.
    :return: None. The function plots the PR curve with confidence intervals.
    """
    num_models = len(boolean_models)
    sample_size = int(sub_ratio * num_models)
    predicted_synergy_scores_list = []

    for i in range(repeat_time):
        if with_seeds:
            np.random.seed(seeds + i)
        sampled_models = np.random.choice(boolean_models, size=sample_size, replace=False).tolist()

        model_predictions = ModelPredictions(
            boolean_models=sampled_models,
            perturbations=perturbations,
            model_outputs=model_outputs,
            synergy_method=synergy_method
        )
        model_predictions.run_simulations(parallel=True)
        predicted_synergy_scores_list.append(model_predictions.predicted_synergy_scores)

    all_predictions = []
    all_observed = []

    for predicted_synergy_scores in predicted_synergy_scores_list:
        df = pd.DataFrame(predicted_synergy_scores, columns=['perturbation', 'synergy_score'])
        df['observed'] = df['perturbation'].apply(lambda x: 1 if x in observed_synergy_scores else 0)
        df['synergy_score'] *= -1
        all_predictions.extend(df['synergy_score'].values)
        all_observed.extend(df['observed'].values)

    pr_df, auc_pr = _calculate_pr_with_ci(
        np.array(all_observed), np.array(all_predictions),
        boot_n, confidence_level, with_seeds, seeds
    )
    PlotUtil.plot_pr_curve_with_ci(pr_df, auc_pr, boot_n=boot_n, plot_discrete=plot_discrete)

def compare_two_simulations(boolean_models1: List, boolean_models2: List, observed_synergy_scores: List[str],
                            model_outputs: Any, perturbations: Any, synergy_method: str = 'hsa',
                            label1: str = 'Models 1', label2: str = 'Models 1',
                            normalized: bool = True) -> None:
    """
    Compares ROC and PR curves for two sets of evolution results.
     By default, normalization of the first result is applied.
    :param boolean_models1: List of the best Boolean Models.
    :param boolean_models2: List of the best Boolean Models.
    :param observed_synergy_scores: List of observed synergy scores for comparison.
    :param model_outputs: Model outputs for evaluation.
    :param perturbations: List of perturbations to apply to the models.
    :param synergy_method: Method to check for synergy ('hsa' or 'bliss').
    :param label1: Label for the evolution_result1.
    :param label2: Label for the evolution_result2.
    :param normalized: Normalize the evolution_result1, True by default.
    :return: None. The function plots the ROC and PR curves.
    """
    predicted_synergy_scores_list = []
    labels = [label1, label2]

    model_predictions1 = ModelPredictions(
        boolean_models=boolean_models1,
        perturbations=perturbations,
        model_outputs=model_outputs,
        synergy_method=synergy_method
    )
    model_predictions1.run_simulations(parallel=True)
    predicted_synergy_scores1 = model_predictions1.predicted_synergy_scores
    predicted_synergy_scores_list.append(predicted_synergy_scores1)

    model_predictions2 = ModelPredictions(
        boolean_models=boolean_models2,
        perturbations=perturbations,
        model_outputs=model_outputs,
        synergy_method=synergy_method
    )
    model_predictions2.run_simulations(parallel=True)
    predicted_synergy_scores2 = model_predictions2.predicted_synergy_scores
    predicted_synergy_scores_list.append(predicted_synergy_scores2)

    if normalized:
        normalized_synergy_scores = _normalize_synergy_scores(predicted_synergy_scores1, predicted_synergy_scores2)
        predicted_synergy_scores_list.append(normalized_synergy_scores)
        labels.append('Calibrated (Normalized)')

    PlotUtil.plot_roc_and_pr_curve(predicted_synergy_scores_list, observed_synergy_scores, synergy_method, labels)
