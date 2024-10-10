import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from pydruglogics.model.ModelPredictions import ModelPredictions
from pydruglogics.utils.Logger import Logger


class Statistics:
    def __init__(self, boolean_models=None, observed_synergy_scores=None, model_outputs=None, perturbations=None,
                 synergy_method='hsa', verbosity=3):
        """
        Initializes the Statistics class.
        :param boolean_models: List of BooleanModel instances.
        :param observed_synergy_scores: List of observed synergy scores.
        :param model_outputs: Model outputs for evaluation.
        :param perturbations: Perturbations to apply to the Boolean Models.
        :param synergy_method: Method to check for synergy ('hsa' or 'bliss').
        """
        self._boolean_models = boolean_models or []
        self._observed_synergy_scores = observed_synergy_scores
        self._model_outputs = model_outputs
        self._perturbations = perturbations
        self._synergy_method = synergy_method
        self._logger = Logger(verbosity)

    def sampling(self, repeat_time=10, sub_ratio=0.8):
        """
        Perform bootstrap sampling on the boolean models, run model predictions, and plot ROC and PR curves.
        :param repeat_time: Number of repeat the sampling process.
        :param sub_ratio: Fraction of models to sample randomly for each iteration.
        """
        num_models = len(self._boolean_models)
        sample_size = int(sub_ratio * num_models)

        predicted_synergy_scores_list = []
        for i in range(repeat_time):
            sampled_models = np.random.choice(self._boolean_models, size=sample_size, replace=False).tolist()
            model_predictions = ModelPredictions(
                boolean_models=sampled_models,
                perturbations=self._perturbations,
                model_outputs=self._model_outputs,
                observed_synergy_scores=self._observed_synergy_scores,
                synergy_method=self._synergy_method,
                verbosity=0
            )
            model_predictions.run_simulations(parallel=True)
            predicted_synergy_scores_list.append(model_predictions.predicted_synergy_scores)

        self.plot_roc_and_pr_curve_multiple(predicted_synergy_scores_list)

    def compare_two_simulations(self, evolution_result1, evolution_result2, label1='Evolution 1 Models',
                                label2='Evolution 2 Models'):
        """
        Compares the ROC and PR Curves of two Evolution results by running predictions on both and plotting.
        :param evolution_result1: List of the best Boolean Models produced by one Evolution run
        :param evolution_result2: List of the best Boolean Models produced by another Evolution run
        """
        predicted_synergy_scores_list = []
        labels = [label1, label2]

        for index, best_boolean_models in enumerate([evolution_result1, evolution_result2]):
            model_predictions = ModelPredictions(
                boolean_models=best_boolean_models,
                perturbations=self._perturbations,
                model_outputs=self._model_outputs,
                observed_synergy_scores=self._observed_synergy_scores,
                synergy_method=self._synergy_method,
                verbosity=0
            )
            model_predictions.run_simulations(parallel=True)
            predicted_synergy_scores_list.append(model_predictions.predicted_synergy_scores)

        self.plot_roc_and_pr_curve_multiple(predicted_synergy_scores_list, labels=labels)

    def plot_roc_and_pr_curve_multiple(self, predicted_synergy_scores_list, labels=None):
        """
        Plot the ROC and PR Curves using multiple predicted synergy scores and the observed synergy combinations.
        :param predicted_synergy_scores_list: List of the perturbation and synergy score.
        :param labels: List of labels for each set of predictions.
        """
        if labels is None:
            labels = [f"Sample {i + 1}" for i in range(len(predicted_synergy_scores_list))]

        plt.figure(figsize=(12, 5))

        # ROC Curve
        plt.subplot(1, 2, 1)
        for idx, (predicted_synergy_scores, label) in enumerate(zip(predicted_synergy_scores_list, labels)):
            df = pd.DataFrame(predicted_synergy_scores, columns=['perturbation', 'synergy_score'])
            df['observed'] = df['perturbation'].apply(lambda x: 1 if x in self._observed_synergy_scores else 0)
            df['synergy_score'] = df['synergy_score'] * -1
            df = df.sort_values(by='synergy_score', ascending=False).reset_index(drop=True)

            self._logger.log(f"Predicted Data with Observed Synergies for {label}:", 1)
            print(df)

            fpr, tpr, _ = roc_curve(df['observed'], df['synergy_score'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{label} AUC: {roc_auc:.2f}")

        plt.plot([0, 1], [0, 1], color='lightgrey', lw=1.2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f"ROC Curve, Ensemble-wise synergies ({self._synergy_method})")
        plt.legend(loc="lower right")
        plt.grid(lw=0.5, color='lightgrey')

        # PR Curve
        plt.subplot(1, 2, 2)
        for idx, (predicted_synergy_scores, label) in enumerate(zip(predicted_synergy_scores_list, labels)):
            df = pd.DataFrame(predicted_synergy_scores, columns=['perturbation', 'synergy_score'])
            df['observed'] = df['perturbation'].apply(lambda x: 1 if x in self._observed_synergy_scores else 0)
            df['synergy_score'] = df['synergy_score'] * -1
            df = df.sort_values(by='synergy_score', ascending=False).reset_index(drop=True)

            # PR Curve
            precision, recall, tresholds = precision_recall_curve(df['observed'], df['synergy_score'])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, label=f"{label} AUC: {pr_auc:.2f}")

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f"PR Curve, Ensemble-wise synergies ({self._synergy_method})")
        plt.grid(lw=0.5, color='lightgrey')
        plt.plot([0, 1], [sum(df['observed']) / len(df['observed'])] * 2, linestyle='--', color='grey')
        plt.legend(loc="upper right")

        plt.tight_layout()
        plt.show()

        # Logger info
        for idx, (predicted_synergy_scores, label) in enumerate(zip(predicted_synergy_scores_list, labels)):
            df = pd.DataFrame(predicted_synergy_scores, columns=['perturbation', 'synergy_score'])
            df['observed'] = df['perturbation'].apply(lambda x: 1 if x in self._observed_synergy_scores else 0)
            df['synergy_score'] = df['synergy_score'] * -1
            fpr, tpr, _ = roc_curve(df['observed'], df['synergy_score'])
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(df['observed'], df['synergy_score'])
            pr_auc = auc(recall, precision)
            self._logger.log(f"ROC AUC for {label}: {roc_auc:.2f}", 3)
            self._logger.log(f"PR AUC for {label}: {pr_auc:.2f}", 3)
