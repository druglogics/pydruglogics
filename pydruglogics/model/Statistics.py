import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pydruglogics.model.ModelPredictions import ModelPredictions
from sklearn.metrics import precision_recall_curve, auc
from pydruglogics.utils.PlotUtil import PlotUtil


class Statistics:
    def __init__(self, boolean_models=None, observed_synergy_scores=None, model_outputs=None, perturbations=None,
                 synergy_method='hsa'):
        """
        Initializes the Statistics class.
        :param boolean_models: List of BooleanModel.
        :param observed_synergy_scores: List of observed synergy scores.
        :param model_outputs: Model outputs for evaluation.
        :param perturbations: List of perturbations to apply to the Boolean Models.
        :param synergy_method: Method to check for synergy ('hsa' or 'bliss').
        """
        self._boolean_models = boolean_models or []
        self._observed_synergy_scores = observed_synergy_scores
        self._model_outputs = model_outputs
        self._perturbations = perturbations
        self._synergy_method = synergy_method

    def _normalize_synergy_scores(self, calibrated_synergy_scores, prolif_synergy_scores):
        normalized_synergy_scores = []
        for (perturbation, ss_score), (_, prolif_score) in zip(calibrated_synergy_scores, prolif_synergy_scores):
            normalized_synergy_score = math.exp(ss_score - prolif_score)
            normalized_synergy_scores.append((perturbation, normalized_synergy_score))

        return normalized_synergy_scores

    def sampling_with_ci(self, repeat_time=10, sub_ratio=0.8, boot_n=1000, confidence_level=0.9,
                         plot_discrete=False, with_seeds=True, seeds=42):
        num_models = len(self._boolean_models)
        sample_size = int(sub_ratio * num_models)
        predicted_synergy_scores_list = []

        for i in range(repeat_time):
            if with_seeds:
                np.random.seed(seeds + i)
            sampled_models = np.random.choice(self._boolean_models, size=sample_size, replace=False).tolist()
            model_predictions = ModelPredictions(
                boolean_models=sampled_models,
                perturbations=self._perturbations,
                model_outputs=self._model_outputs,
                synergy_method=self._synergy_method
            )
            model_predictions.run_simulations(parallel=True)
            predicted_synergy_scores_list.append(model_predictions.predicted_synergy_scores)

        all_predictions = []
        all_observed = []

        for predicted_synergy_scores in predicted_synergy_scores_list:
            df = pd.DataFrame(predicted_synergy_scores, columns=['perturbation', 'synergy_score'])
            df['observed'] = df['perturbation'].apply(lambda x: 1 if x in self._observed_synergy_scores else 0)
            df['synergy_score'] = df['synergy_score'] * -1
            all_predictions.extend(df['synergy_score'].values)
            all_observed.extend(df['observed'].values)

        all_observed = np.array(all_observed)
        all_predictions = np.array(all_predictions)

        pr_df, auc_pr = self.calculate_pr_with_ci(all_observed, all_predictions, boot_n=boot_n,
                                                       confidence_level=confidence_level,
                                                       with_seeds=with_seeds, seeds=seeds)

        PlotUtil.plot_pr_curve_with_ci(pr_df, auc_pr, boot_n=boot_n, plot_discrete=plot_discrete)

    def calculate_pr_with_ci(self, observed, preds, boot_n, confidence_level, with_seeds, seeds):
        if with_seeds:
            np.random.seed(seeds)

        precision_orig, recall_orig, _ = precision_recall_curve(observed, preds)
        auc_pr = auc(recall_orig, precision_orig)

        pr_df = pd.DataFrame({'recall': recall_orig, 'precision': precision_orig})

        resampled_data = self.bootstrap_resample(observed, preds, boot_n=boot_n)
        precision_matrix = []

        for resampled_observed, resampled_predicted in resampled_data:
            precision_boot, recall_boot, _ = precision_recall_curve(resampled_observed, resampled_predicted)
            const_interp_pr = interp1d(recall_boot, precision_boot, kind='previous', bounds_error=False,
                                      fill_value=(precision_boot[0], precision_boot[-1]))
            aligned_precisions = const_interp_pr(recall_orig)
            precision_matrix.append(aligned_precisions)

        precision_matrix = np.array(precision_matrix)

        alpha = 1-confidence_level

        low_precision = np.percentile(precision_matrix, alpha / 2 * 100, axis=0)
        high_precision = np.percentile(precision_matrix, (1 - alpha / 2) * 100, axis=0)

        pr_df['low_precision'] = low_precision
        pr_df['high_precision'] = high_precision

        return pr_df, auc_pr

    def compare_two_simulations(self, evolution_result1, evolution_result2, label1='Evolution 1 Models',
                                label2='Evolution 2 Models', normalized=True):
        """
        Compares the ROC and PR Curves of two Evolution results (list of the best Boolean Models).
        By default normalization of the first result is true.
        :param evolution_result1: List of the best Boolean Models.
        :param evolution_result2: List of the best Boolean Models.
        :param label1: Label for the evolution_result1.
        :param label2: Label for the evolution_result2.
        :param normalized: Normalize the evolution_result1, True by default.
        """
        predicted_synergy_scores_list = []
        labels = [label1, label2]

        model_predictions1 = ModelPredictions(
            boolean_models=evolution_result1,
            perturbations=self._perturbations,
            model_outputs=self._model_outputs,
            synergy_method=self._synergy_method
        )
        model_predictions1.run_simulations(parallel=True)
        predicted_synergy_scores1 = model_predictions1.predicted_synergy_scores
        predicted_synergy_scores_list.append(predicted_synergy_scores1)

        model_predictions2 = ModelPredictions(
            boolean_models=evolution_result2,
            perturbations=self._perturbations,
            model_outputs=self._model_outputs,
            synergy_method=self._synergy_method
        )
        model_predictions2.run_simulations(parallel=True)
        predicted_synergy_scores2 = model_predictions2.predicted_synergy_scores
        predicted_synergy_scores_list.append(predicted_synergy_scores2)

        if normalized:
            normalized_synergy_scores = self._normalize_synergy_scores(predicted_synergy_scores1,
                                                                       predicted_synergy_scores2)
            predicted_synergy_scores_list.append(normalized_synergy_scores)
            labels.append('Calibrated (Normalized)')

        PlotUtil.plot_roc_and_pr_curve(predicted_synergy_scores_list,
                                       self._observed_synergy_scores, self._synergy_method, labels)

    def bootstrap_resample(self, labels, predictions, boot_n):
        resampled_model_preds = []
        for _ in range(boot_n):
            rnd = np.random.choice(len(labels), size=len(labels), replace=True)
            resampled_labels = labels[rnd]
            resampled_predictions = predictions[rnd]
            resampled_model_preds.append((resampled_labels, resampled_predictions))
        return resampled_model_preds
