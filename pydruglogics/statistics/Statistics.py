import logging
import math
import os
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm
from sklearn.metrics import precision_recall_curve, auc
from pydruglogics.model.ModelPredictions import ModelPredictions
from pydruglogics.utils.PlotUtil import PlotUtil
from typing import List, Tuple, Any

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
                          confidence_level: float, with_seeds: bool, seeds: int) -> Tuple[pd.DataFrame, float, dict]:
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

    sample_mean = np.mean(precision_orig)
    standard_deviation = np.std(precision_orig)
    standard_err = standard_deviation/np.sqrt(len(precision_orig))
    critical_value = norm.ppf(1-alpha/ 2)
    margin_of_err = critical_value * standard_err
    ci_lower = float(sample_mean - margin_of_err)
    ci_upper = float(sample_mean + margin_of_err)

    sampl_with_ci_results= {
        "Point Estimate (Mean)": sample_mean,
        "Standard Deviation": standard_deviation,
        "Standard Error": standard_err,
        "Confidence Interval": (ci_lower, ci_upper),
        "Confidence Level": f"{confidence_level * 100}%",
        "Critical Value": critical_value,
        "Margin of Error": margin_of_err,
        "Sample Size": len(precision_orig)
    }

    return pr_df, auc_pr, sampl_with_ci_results


def _create_result_base_folder(main_folder: str, category_folder: str, prefix: str) -> str:

    base_path = os.path.join(main_folder, category_folder)
    os.makedirs(base_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M')
    result_folder = os.path.join(base_path, f"{prefix}_{timestamp}")
    os.makedirs(result_folder, exist_ok=True)

    return result_folder


def _save_sampling_results(synergy_scores: List[List[Tuple[str, float]]], base_folder: str, synergy_method: str,
                           summary_metrics: dict):
    current_time = datetime.now().strftime('%Y/%m/%d %H:%M')
    summary_path = os.path.join(base_folder, "sampling_with_ci.tab")

    with open(summary_path, 'w') as file:
        file.write(f"Date: {current_time.split()[0]}, Time: {current_time.split()[1]}\n")
        file.write("Sampling Results\n")
        for key, value in summary_metrics.items():
            formatted_value = f"({value[0]:.6f}, {value[1]:.6f})" if isinstance(value, tuple) else f"{value:.6f}" if isinstance(value, float) else str(value)
            file.write(f"{key}: {formatted_value}\n")

    for idx, sample in enumerate(synergy_scores, start=1):
        file_path = os.path.join(base_folder, f"synergy_scores_sample_{idx}.tab")
        with open(file_path, 'w') as file:
            file.write(f"# Date: {current_time.split()[0]}, Time: {current_time.split()[1]}\n")
            file.write(f"# Synergies ({synergy_method})\n")
            file.write("perturbation_name\tsynergy_score\n")
            for perturbation, score in sample:
                file.write(f"{perturbation}\t{score}\n")

    logging.info(f"Sampling results saved to {base_folder}")


def _save_compare_results(predicted_synergy_scores_list: List[List[Tuple[str, float]]], labels: List[str],
                          base_folder: str, synergy_method: str):
    current_time = datetime.now().strftime('%Y/%m/%d %H:%M')

    for scores, label in zip(predicted_synergy_scores_list, labels):
        underscore_label = label.replace(" ", "_").lower()
        file_path = os.path.join(base_folder, f"predicted_synergy_scores_{underscore_label}.tab")

        with open(file_path, 'w') as file:
            file.write(f"# Date: {current_time.split()[0]}, Time: {current_time.split()[1]}\n")
            file.write(f"# Synergies ({synergy_method})\n")
            file.write("perturbation_name\tsynergy_score\n")
            for perturbation, score in scores:
                file.write(f"{perturbation}\t{score}\n")

    logging.info(f"Comparison results saved to {base_folder}")

def sampling_with_ci(boolean_models: List, observed_synergy_scores: List[str], model_outputs: Any,
                     perturbations: Any, synergy_method: str = 'bliss', repeat_time: int = 10, sub_ratio: float = 0.8,
                     boot_n: int = 1000, confidence_level: float = 0.9, plot: bool = True, plot_discrete: bool = False,
                     save_result: bool = True, with_seeds: bool = True, seeds: int = 42) -> None:
    """
    Performs sampling with confidence interval calculation and plot the PR curve.
    :param boolean_models: List of BooleanModel instances.
    :param observed_synergy_scores: List of observed synergy scores.
    :param model_outputs: Model outputs for evaluation.
    :param perturbations: List of perturbations to apply to the models.
    :param synergy_method: Method to check for synergy ('hsa' or 'bliss').
    :param repeat_time: Number of times to repeat sampling.
    :param sub_ratio: Proportion of models to sample in each iteration.
    :param boot_n: Number of bootstrap resampling iterations for confidence intervals.
    :param confidence_level: Confidence level for confidence interval calculations.
    :param plot: Whether to display the PR curve.
    :param plot_discrete: Whether to plot discrete points on the PR curve.
    :param save_result: Whether to save the results to a .tab file.
    :param with_seeds: Whether to use a fixed seed for reproducibility.
    :param seeds: Seed value for random number generation to ensure reproducibility.
    :return: None
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

    for pred_synergy_scores in predicted_synergy_scores_list:
        df = pd.DataFrame(pred_synergy_scores, columns=['perturbation', 'synergy_score'])
        df['observed'] = df['perturbation'].apply(lambda pert: 1 if pert in observed_synergy_scores else 0)
        df['synergy_score'] *= -1
        all_predictions.extend(df['synergy_score'].values)
        all_observed.extend(df['observed'].values)

    pr_df, auc_pr, summary_metrics = _calculate_pr_with_ci(np.array(all_observed), np.array(all_predictions),
                                                           boot_n, confidence_level, with_seeds, seeds)

    if save_result:
        base_folder = _create_result_base_folder('results', 'sampling', 'sampling')
        _save_sampling_results(synergy_scores=predicted_synergy_scores_list, base_folder=base_folder,
                               synergy_method=synergy_method, summary_metrics=summary_metrics)

    if plot:
        PlotUtil.plot_pr_curve_with_ci(pr_df, auc_pr, boot_n=boot_n, plot_discrete=plot_discrete)


def compare_two_simulations(boolean_models1: List, boolean_models2: List, observed_synergy_scores: List[str],
                            model_outputs: Any, perturbations: Any, synergy_method: str = 'bliss',
                            label1: str = 'Models 1', label2: str = 'Models 2', normalized: bool = True,
                            plot: bool = True, save_result: bool = True) -> None:
    """
    Compares ROC and PR curves for two sets of evolution results.
    By default, normalization of the first result is applied.
    :param boolean_models1: List of the best Boolean Models for the first simulation set.
    :param boolean_models2: List of the best Boolean Models for the second simulation set.
    :param observed_synergy_scores: List of observed synergy scores for comparison.
    :param model_outputs: Model outputs for evaluation.
    :param perturbations: List of perturbations to apply to the models.
    :param synergy_method: Method to check for synergy ('hsa' or 'bliss').
    :param label1: Label for the first simulation result.
    :param label2: Label for the second simulation result.
    :param normalized: Whether to normalize the first result.
    :param plot: Whether to display the ROC and PR curves.
    :param save_result: Whether to save the results.
    :return: None
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

    if save_result:
        base_folder = _create_result_base_folder('results', 'comparison', 'comparison')
        _save_compare_results(predicted_synergy_scores_list, labels, base_folder=base_folder, synergy_method=synergy_method)

    if plot:
        PlotUtil.plot_roc_and_pr_curve(predicted_synergy_scores_list, observed_synergy_scores, synergy_method, labels)
