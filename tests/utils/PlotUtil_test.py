import logging
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pydruglogics.utils.PlotUtil import PlotUtil
from matplotlib import pyplot as plt

class TestPlotUtil:

    @patch("pydruglogics.utils.PlotUtil.plt.show")
    @patch("pydruglogics.utils.PlotUtil.plt.step")
    @patch("pydruglogics.utils.PlotUtil.plt.fill_between")
    @patch("pydruglogics.utils.PlotUtil.plt.vlines")
    def test_plot_pr_curve_with_ci_discrete(self, mock_vlines, mock_fill_between, mock_step, mock_show):
        pr_df = pd.DataFrame({
            'recall': [0, 0.5, 1],
            'precision': [0.8, 0.6, 0.4],
            'low_precision': [0.7, 0.5, 0.6],
            'high_precision': [0.9, 0.7, 0.8]
        })
        auc_pr = 0.80
        boot_n = 1000
        PlotUtil.plot_pr_curve_with_ci(pr_df, auc_pr, boot_n, plot_discrete=True)
        mock_step.assert_called_once_with(pr_df['recall'], pr_df['precision'], color='red',
                                          label='Precision-Recall curve', where='post')
        assert mock_vlines.call_count == len(pr_df)
        mock_show.assert_called_once()

    @patch("pydruglogics.utils.PlotUtil.plt.show")
    @patch("pydruglogics.utils.PlotUtil.plt.fill_between")
    def test_plot_pr_curve_with_ci_continuous(self, mock_fill_between, mock_show):
        pr_df = pd.DataFrame({
            'recall': [0, 0.5, 1],
            'precision': [0.8, 0.6, 0.4],
            'low_precision': [0.7, 0.5, 0.6],
            'high_precision': [0.9, 0.7, 0.8]
        })
        auc_pr = 0.75
        boot_n = 1000
        PlotUtil.plot_pr_curve_with_ci(pr_df, auc_pr, boot_n, plot_discrete=False)
        mock_fill_between.assert_called_once_with(pr_df['recall'], pr_df['low_precision'],
                                                  pr_df['high_precision'], color='grey', alpha=0.3,
                                                  label='Confidence Interval')
        mock_show.assert_called_once()

    @patch("pydruglogics.utils.PlotUtil.plt.show")
    @patch("pydruglogics.utils.PlotUtil.plt.plot")
    @patch("pydruglogics.utils.PlotUtil.plt.legend")
    def test_plot_roc_and_pr_curve_single_model(self, mock_legend, mock_plot, mock_show):
        predicted_synergy_scores = [[
            {'perturbation': 'A', 'synergy_score': 0.9},
            {'perturbation': 'B', 'synergy_score': 0.7},
            {'perturbation': 'C', 'synergy_score': 0.4}
        ]]
        observed_synergy_scores = ['A', 'C']
        synergy_method = "Test Method"
        PlotUtil.plot_roc_and_pr_curve(predicted_synergy_scores, observed_synergy_scores, synergy_method)
        assert mock_plot.call_count > 0
        mock_show.assert_called_once()

    @patch("pydruglogics.utils.PlotUtil.plt.show")
    @patch("pydruglogics.utils.PlotUtil.plt.plot")
    @patch("pydruglogics.utils.PlotUtil.plt.legend")
    def test_plot_roc_and_pr_curve_multiple_models(self, mock_legend, mock_plot, mock_show):
        predicted_synergy_scores = [
            [
                {'perturbation': 'A', 'synergy_score': 0.9},
                {'perturbation': 'B', 'synergy_score': 0.6}
            ],
            [
                {'perturbation': 'A', 'synergy_score': 0.85},
                {'perturbation': 'B', 'synergy_score': 0.75}
            ]
        ]
        observed_synergy_scores = ['A']
        synergy_method = "Test Method"
        labels = ["Model 1", "Model 2"]
        PlotUtil.plot_roc_and_pr_curve(predicted_synergy_scores, observed_synergy_scores, synergy_method, labels=labels)
        assert mock_plot.call_count > 0
        mock_show.assert_called_once()

    @patch("pydruglogics.utils.PlotUtil.plt.show")
    @patch("pydruglogics.utils.PlotUtil.plt.plot")
    @patch("pydruglogics.utils.PlotUtil.plt.legend")
    def test_plot_roc_and_pr_curve_default_labels(self, mock_legend, mock_plot, mock_show):
        predicted_synergy_scores = [
            [
                {'perturbation': 'A', 'synergy_score': 0.9},
                {'perturbation': 'B', 'synergy_score': 0.6}
            ],
            [
                {'perturbation': 'A', 'synergy_score': 0.85},
                {'perturbation': 'B', 'synergy_score': 0.75}
            ]
        ]
        observed_synergy_scores = ['A']
        synergy_method = "Test Method"
        PlotUtil.plot_roc_and_pr_curve(predicted_synergy_scores, observed_synergy_scores, synergy_method)
        assert mock_plot.call_count > 0
        mock_show.assert_called_once()

    @patch("pydruglogics.utils.PlotUtil.plt.show")
    @patch("pandas.DataFrame.sort_values")
    @patch("pandas.DataFrame.apply")
    def test_data_transformation_and_logging(self, mock_apply, mock_sort_values, mock_show):
        mock_apply.side_effect = lambda x: pd.Series([1 if pert in ['A', 'C'] else 0 for pert in x])
        mock_sort_values.return_value = pd.DataFrame({
            'perturbation': ['A', 'B', 'C'],
            'synergy_score': [-0.9, -0.6, -0.4],
            'observed': [1, 0, 1]
        })

        predicted_synergy_scores = [[
            {'perturbation': 'A', 'synergy_score': 0.9},
            {'perturbation': 'B', 'synergy_score': 0.7},
            {'perturbation': 'C', 'synergy_score': 0.4}
        ]]
        observed_synergy_scores = ['A', 'C']
        synergy_method = "Test Method"

        with patch.object(logging, 'info') as mock_info:
            PlotUtil.plot_roc_and_pr_curve(predicted_synergy_scores, observed_synergy_scores, synergy_method)
            mock_info.assert_any_call("Predicted Data with Observed Synergies for Model 1:")

    @patch("pydruglogics.utils.PlotUtil.plt.show")
    @patch("pydruglogics.utils.PlotUtil.plt.plot")
    def test_plot_roc_and_pr_curve_single_model_wrapping(self, mock_plot, mock_show):
        predicted_synergy_scores = [
            {'perturbation': 'A', 'synergy_score': 0.9},
            {'perturbation': 'B', 'synergy_score': 0.7},
            {'perturbation': 'C', 'synergy_score': 0.4}
        ]
        observed_synergy_scores = ['A', 'C']
        synergy_method = "Test Method"
        PlotUtil.plot_roc_and_pr_curve(predicted_synergy_scores, observed_synergy_scores, synergy_method)
        assert mock_plot.call_count > 0
        mock_show.assert_called_once()
