import math

import pytest
import numpy as np
import pandas as pd
import os
from unittest.mock import patch, Mock, call
from pydruglogics.statistics.Statistics import (_normalize_synergy_scores, _bootstrap_resample, _calculate_pr_with_ci,
                                                _create_result_base_folder, _save_sampling_results,
                                                _save_compare_results, sampling_with_ci, compare_two_simulations)
from pydruglogics.model.ModelPredictions import ModelPredictions


class TestStatistics:

    @pytest.fixture
    def sample_synergy_scores(self):
        calibrated_scores = [('SC1', 1.5), ('SC2', 2.0), ('SC3', 1.8)]
        proliferative_scores = [('SC1', 1.0), ('SC2', 1.5), ('SC3', 1.2)]
        return calibrated_scores, proliferative_scores

    @pytest.fixture
    def pr_ci_data(self):
        observed = np.array([1, 0, 1, 0, 1])
        preds = np.array([0.9, 0.1, 0.8, 0.3, 0.7])
        return observed, preds

    @pytest.fixture
    def boolean_models(self):
        return [Mock() for _ in range(10)]

    @pytest.fixture
    def observed_synergy_scores(self):
        return ['SC1', 'SC3']

    @pytest.fixture
    def model_outputs(self):
        return Mock()

    @pytest.fixture
    def perturbations(self):
        return Mock()

    def test_normalize_synergy_scores(self, sample_synergy_scores):
        calibrated_scores, proliferative_scores = sample_synergy_scores
        result = _normalize_synergy_scores(calibrated_scores, proliferative_scores)
        expected = [
            ('SC1', math.exp(1.5 - 1.0)),
            ('SC2', math.exp(2.0 - 1.5)),
            ('SC3', math.exp(1.8 - 1.2))
        ]

        assert len(result) == len(expected)
        for (res_name, res_value), (exp_name, exp_value) in zip(result, expected):
            assert res_name == exp_name
            assert res_value == pytest.approx(exp_value)

    def test_calculate_pr_with_ci_seed(self, pr_ci_data):
        observed, preds = pr_ci_data
        with patch('numpy.random.seed') as mock_seed:
            pr_df, auc_pr, summary_metrics = _calculate_pr_with_ci(
                observed, preds, boot_n=100, confidence_level=0.95, with_seeds=True, seeds=42
            )
            mock_seed.assert_called_once_with(42)
            assert isinstance(pr_df, pd.DataFrame)
            assert 'recall' in pr_df.columns
            assert 'precision' in pr_df.columns
            assert 'low_precision' in pr_df.columns
            assert 'high_precision' in pr_df.columns

    def test_calculate_pr_with_ci_no_seed(self, pr_ci_data):
        observed, preds = pr_ci_data
        with patch('numpy.random.seed') as mock_seed:
            pr_df, auc_pr, summary_metrics = _calculate_pr_with_ci(
                observed, preds, boot_n=100, confidence_level=0.95, with_seeds=False, seeds=42
            )
            mock_seed.assert_not_called()

    def test_create_result_base_folder(self, tmp_path):
        main_folder = tmp_path / "results"
        category_folder = "sampling"
        prefix = "test"
        result_folder = _create_result_base_folder(main_folder, category_folder, prefix)

        assert os.path.exists(result_folder)
        assert result_folder.startswith(str(main_folder / category_folder / f"{prefix}_"))

    @patch('builtins.open')
    @patch('pydruglogics.statistics.Statistics.datetime')
    def test_save_sampling_results(self, mock_datetime, mock_open, tmp_path):
        mock_datetime.now.return_value.strftime.return_value = "2024/11/16 12:30"
        synergy_scores = [[('SC1', 1.2), ('SC2', 1.5)], [('SC3', 1.8), ('SC4', 2.0)]]
        base_folder = tmp_path / "test_results"
        summary_metrics = {"AUC-PR": 0.85, "Confidence Interval": (0.80, 0.90)}

        _save_sampling_results(synergy_scores, base_folder, "bliss", summary_metrics)

        assert mock_open.call_count == 3
        mock_open.assert_any_call(os.path.join(base_folder, "sampling_with_ci.tab"), 'w')

    @patch('builtins.open')
    @patch('pydruglogics.statistics.Statistics.datetime')
    def test_save_compare_results(self, mock_datetime, mock_open, tmp_path):
        mock_datetime.now.return_value.strftime.return_value = "2024/11/16 12:30"
        predicted_synergy_scores_list = [[('SC1', 1.2), ('SC2', 1.5)], [('SC3', 1.8), ('SC4', 2.0)]]
        labels = ['Model A', 'Model B']
        base_folder = tmp_path / "comparison_results"

        _save_compare_results(predicted_synergy_scores_list, labels, base_folder, "hsa")

        assert mock_open.call_count == 2
        mock_open.assert_any_call(os.path.join(base_folder, "predicted_synergy_scores_model_a.tab"), 'w')
        mock_open.assert_any_call(os.path.join(base_folder, "predicted_synergy_scores_model_b.tab"), 'w')

    @patch('pydruglogics.statistics.Statistics._create_result_base_folder', return_value='mock_base_folder')
    @patch('pydruglogics.statistics.Statistics._save_sampling_results')
    @patch.object(ModelPredictions, 'run_simulations')
    @patch('pydruglogics.statistics.Statistics.PlotUtil.plot_pr_curve_with_ci')
    def test_sampling_with_ci_with_seed(self, mock_plot, mock_run, mock_save, mock_base_folder, boolean_models,
                                        observed_synergy_scores, model_outputs, perturbations):
        mock_run.side_effect = lambda *args, **kwargs: setattr(ModelPredictions,
                                                               'predicted_synergy_scores',
                                                               [('SC1', -0.5), ('SC2', -1.2), ('SC3', -0.9),
                                                                ('SC4', -1.1), ('SC5', -0.7)])

        sampling_with_ci(
            boolean_models, observed_synergy_scores, model_outputs, perturbations,
            repeat_time=2, boot_n=50, with_seeds=True, seeds=100
        )

        mock_run.assert_called()
        mock_save.assert_called_once()
        mock_plot.assert_called_once()

    def test_sampling_with_ci_no_seed(self, boolean_models, observed_synergy_scores, model_outputs, perturbations):
        with patch('pydruglogics.statistics.Statistics._create_result_base_folder') as mock_base_folder, \
                patch('pydruglogics.statistics.Statistics._save_sampling_results') as mock_save, \
                patch.object(ModelPredictions, 'run_simulations') as mock_run, \
                patch('pydruglogics.statistics.Statistics.PlotUtil.plot_pr_curve_with_ci') as mock_plot:
            mock_base_folder.return_value = "mock_base_folder"
            sampling_with_ci(boolean_models, observed_synergy_scores, model_outputs, perturbations,
                             repeat_time=2, boot_n=50, with_seeds=False)

            mock_run.assert_called()
            mock_save.assert_called_once()
            mock_plot.assert_called_once()


    @patch('pydruglogics.statistics.Statistics._create_result_base_folder', return_value='mock_base_folder')
    @patch('pydruglogics.statistics.Statistics._save_compare_results')
    @patch.object(ModelPredictions, 'run_simulations')
    @patch('pydruglogics.statistics.Statistics._normalize_synergy_scores')
    @patch('pydruglogics.statistics.Statistics.PlotUtil.plot_roc_and_pr_curve')
    def test_compare_two_simulations(self, mock_plot, mock_normalize, mock_run, mock_save, mock_base_folder,
                                     boolean_models, observed_synergy_scores, model_outputs, perturbations):
        mock_run.side_effect = lambda *args, **kwargs: setattr(ModelPredictions, 'predicted_synergy_scores',
                                                               [('SC1', 1.0), ('SC2', 0.5), ('SC3', 1.5)])
        mock_normalize.return_value = [('SC1', 1.2), ('SC2', 1.1), ('SC3', 1.0)]

        compare_two_simulations(
            boolean_models1=boolean_models,
            boolean_models2=boolean_models,
            observed_synergy_scores=observed_synergy_scores,
            model_outputs=model_outputs,
            perturbations=perturbations,
            synergy_method='bliss',
            label1='Model 1',
            label2='Model 2',
            normalized=True,
            plot=True,
            save_result=True)

        mock_run.assert_called()
        mock_normalize.assert_called_once()
        mock_save.assert_called_once_with([[('SC1', 1.0), ('SC2', 0.5), ('SC3', 1.5)],
                                           [('SC1', 1.0), ('SC2', 0.5), ('SC3', 1.5)],
                                           [('SC1', 1.2), ('SC2', 1.1), ('SC3', 1.0)]],
                                          ['Model 1', 'Model 2', 'Calibrated (Normalized)'],
                                          base_folder='mock_base_folder',synergy_method='bliss')

        mock_plot.assert_called_once_with([[('SC1', 1.0), ('SC2', 0.5), ('SC3', 1.5)],
                                           [('SC1', 1.0), ('SC2', 0.5), ('SC3', 1.5)],
                                           [('SC1', 1.2), ('SC2', 1.1), ('SC3', 1.0)]],
                                          observed_synergy_scores,'bliss',['Model 1', 'Model 2',
                                                                           'Calibrated (Normalized)'])