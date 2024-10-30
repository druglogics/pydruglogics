import pandas as pd
import logging
from matplotlib import pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve


class PlotUtil:

    @staticmethod
    def plot_roc_and_pr_curve(predicted_synergy_scores, observed_synergy_scores, synergy_method, labels=None):
        """
        Plot the ROC and PR Curves for one or multiple sets of predicted synergy scores.
        :param predicted_synergy_scores: Either a list of predictions for multiple models or a single set of predictions.
        :param observed_synergy_scores: List of observed synergy scores.
        :param synergy_method: Method used for synergy scoring (for plot titles).
        :param labels: Optional list of labels for each set of predictions. If None, default labels will be generated.
        """

        single_model = not isinstance(predicted_synergy_scores[0], list)
        if single_model:
            predicted_synergy_scores = [predicted_synergy_scores]

        if labels is None:
            labels = [f"Model {i + 1}" for i in range(len(predicted_synergy_scores))]

        plt.figure(figsize=(12, 5))

        # ROC Curve
        plt.subplot(1, 2, 1)
        for idx, (predicted_scores, label) in enumerate(zip(predicted_synergy_scores, labels)):
            df = pd.DataFrame(predicted_scores, columns=['perturbation', 'synergy_score'])
            df['observed'] = df['perturbation'].apply(lambda x: 1 if x in observed_synergy_scores else 0)
            df['synergy_score'] = df['synergy_score'] * -1
            df = df.sort_values(by='synergy_score', ascending=False).reset_index(drop=True)

            logging.info(f"Predicted Data with Observed Synergies for {label}:")
            logging.info(df)

            fpr, tpr, _ = roc_curve(df['observed'], df['synergy_score'])
            roc_auc = auc(fpr, tpr)
            auc_label = f"AUC: {roc_auc:.2f}" if single_model else f"{label} AUC: {roc_auc:.2f}"
            plt.plot(fpr, tpr, lw=2, label=auc_label)

        plt.plot([0, 1], [0, 1], color='lightgrey', lw=1.2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f"ROC Curve, Ensemble-wise synergies ({synergy_method})")
        plt.legend(loc="lower right")
        plt.grid(lw=0.5, color='lightgrey')

        # PR Curve
        plt.subplot(1, 2, 2)
        for idx, (predicted_scores, label) in enumerate(zip(predicted_synergy_scores, labels)):
            df = pd.DataFrame(predicted_scores, columns=['perturbation', 'synergy_score'])
            df['observed'] = df['perturbation'].apply(lambda x: 1 if x in observed_synergy_scores else 0)
            df['synergy_score'] = df['synergy_score'] * -1
            df = df.sort_values(by='synergy_score', ascending=False).reset_index(drop=True)

            precision, recall, _ = precision_recall_curve(df['observed'], df['synergy_score'])
            pr_auc = auc(recall, precision)
            auc_label = f"AUC: {pr_auc:.2f}" if single_model else f"{label} AUC: {pr_auc:.2f}"
            plt.plot(recall, precision, lw=2, label=auc_label)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f"PR Curve, Ensemble-wise synergies ({synergy_method})")
        plt.grid(lw=0.5, color='lightgrey')
        plt.plot([0, 1], [sum(df['observed']) / len(df['observed'])] * 2, linestyle='--', color='grey')
        plt.legend(loc="upper right")

        plt.tight_layout()
        plt.show()

        # Logging
        for idx, (predicted_scores, label) in enumerate(zip(predicted_synergy_scores, labels)):
            df = pd.DataFrame(predicted_scores, columns=['perturbation', 'synergy_score'])
            df['observed'] = df['perturbation'].apply(lambda x: 1 if x in observed_synergy_scores else 0)
            df['synergy_score'] = df['synergy_score'] * -1
            fpr, tpr, _ = roc_curve(df['observed'], df['synergy_score'])
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(df['observed'], df['synergy_score'])
            pr_auc = auc(recall, precision)
            logging.debug(f"ROC AUC for {label}: {roc_auc:.2f}")
            logging.debug(f"PR AUC for {label}: {pr_auc:.2f}")


    @staticmethod
    def plot_pr_curve_with_ci(pr_df, auc_pr, boot_n, plot_discrete):
        plt.figure(figsize=(8, 6))

        if plot_discrete:
            plt.step(pr_df['recall'], pr_df['precision'], color='red', label='Precision-Recall curve', where='post')
            plt.plot(pr_df['recall'], pr_df['low_precision'], linestyle='--', color='grey',
                     label='Lower Confidence Interval', marker='s')
            plt.plot(pr_df['recall'], pr_df['high_precision'], linestyle='--', color='grey',
                     label='Upper Confidence Interval', marker='s')
            for i in range(len(pr_df)):
                plt.vlines(x=pr_df['recall'].iloc[i], ymin=pr_df['low_precision'].iloc[i],
                           ymax=pr_df['high_precision'].iloc[i], alpha=0.3,
                           color='grey', linestyle='dotted')
        else:
            plt.plot(pr_df['recall'], pr_df['precision'], color='red', label='Precision-Recall curve')
            plt.fill_between(pr_df['recall'], pr_df['low_precision'], pr_df['high_precision'], color='grey',
                             alpha=0.3, label='Confidence Interval')

        plt.plot([0, 1], [0.2, 0.2], linestyle='--', color='grey')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim(0.0, 1.1)
        plt.title(f'PR Curve with {boot_n} Samples')
        plt.legend(loc='lower left', title=f'AUC: {auc_pr:.3f}')
        plt.grid(lw=0.5, color='lightgrey')
        plt.show()
