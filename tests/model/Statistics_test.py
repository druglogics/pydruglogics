import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock
from pydruglogics.model.Statistics import (_normalize_synergy_scores, _bootstrap_resample, _calculate_pr_with_ci,
                                            sampling_with_ci, compare_two_simulations)
from pydruglogics.model.ModelPredictions import ModelPredictions
from pydruglogics.utils.PlotUtil import PlotUtil


class TestStatistics:

    @pytest.fixture
    def sample_synergy_scores(self):
        calibrated_scores = [('SC1', 1.5), ('SC2', 2.0), ('SC3', 1.8)]
        proliferative_scores = [('SC1', 1.0), ('SC2', 1.5), ('SC3', 1.2)]
        return calibrated_scores, proliferative_scores

    def test_normalize_synergy_scores(self, sample_synergy_scores):
        calibrated_scores, proliferative_scores = sample_synergy_scores
        result = _normalize_synergy_scores(calibrated_scores, proliferative_scores)
        expected = [('SC1', np.exp(0.5)), ('SC2', np.exp(0.5)), ('SC3', np.exp(0.6))]

        for (res_name, res_value), (exp_name, exp_value) in zip(result, expected):
            assert res_name == exp_name
            assert res_value == pytest.approx(exp_value)

    def test_bootstrap_resample(self):
        labels = np.array([1, 0, 1, 1, 0])
        predictions = np.array([0.9, 0.2, 0.8, 0.7, 0.3])
        boot_n = 3

        result = _bootstrap_resample(labels, predictions, boot_n)
        assert len(result) == boot_n
        for resampled_labels, resampled_preds in result:
            assert len(resampled_labels) == len(labels)
            assert len(resampled_preds) == len(predictions)

    @pytest.fixture
    def pr_ci_data(self):
        observed = np.array([1, 0, 1, 0, 1])
        preds = np.array([0.9, 0.1, 0.8, 0.3, 0.7])
        return observed, preds

    def test_calculate_pr_with_ci(self, pr_ci_data):
        observed, preds = pr_ci_data
        boot_n = 100
        confidence_level = 0.9

        with patch('pydruglogics.model.Statistics._bootstrap_resample', wraps=_bootstrap_resample):
            pr_df, auc_pr = _calculate_pr_with_ci(observed, preds, boot_n, confidence_level, with_seeds=True, seeds=42)

        assert isinstance(pr_df, pd.DataFrame)
        assert 'recall' in pr_df.columns
        assert 'precision' in pr_df.columns
        assert 'low_precision' in pr_df.columns
        assert 'high_precision' in pr_df.columns
        assert isinstance(auc_pr, float)
        assert 0 <= auc_pr <= 1

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

    def test_sampling_with_ci(self, boolean_models, observed_synergy_scores, model_outputs, perturbations):
        with patch.object(ModelPredictions, 'run_simulations') as mock_run_simulations, \
                patch.object(PlotUtil, 'plot_pr_curve_with_ci') as mock_plot, \
                patch('pydruglogics.model.Statistics._calculate_pr_with_ci',
                      wraps=_calculate_pr_with_ci) as mock_calculate_pr:
            mock_run_simulations.return_value = None
            mock_plot.return_value = None
            mock_calculate_pr.return_value = (pd.DataFrame({'recall': [0.0, 0.5, 1.0],'precision': [1.0, 0.75, 0.5],
                                                            'low_precision': [0.8, 0.7, 0.4],
                                                            'high_precision': [1.0, 0.8, 0.6]}), 0.85)

            sampling_with_ci(boolean_models, observed_synergy_scores, model_outputs, perturbations,
                             synergy_method='hsa', repeat_time=5, sub_ratio=0.8, boot_n=100, confidence_level=0.9,
                             plot_discrete=False, with_seeds=True, seeds=42)

            mock_run_simulations.assert_called()
            mock_plot.assert_called()
            assert mock_calculate_pr.called

            pr_df, auc_pr = mock_calculate_pr.return_value
            assert 'recall' in pr_df.columns
            assert 'precision' in pr_df.columns
            assert 'low_precision' in pr_df.columns
            assert 'high_precision' in pr_df.columns
            assert isinstance(auc_pr, float)

    def test_compare_two_simulations(self):
        evolution_result1 = [Mock() for _ in range(3)]
        evolution_result2 = [Mock() for _ in range(3)]
        observed_synergy_scores = ['SC1', 'SC3']
        model_outputs = Mock()
        perturbations = Mock()

        with patch.object(ModelPredictions, 'run_simulations') as mock_run_simulations, \
                patch.object(PlotUtil, 'plot_roc_and_pr_curve') as mock_plot, \
                patch('pydruglogics.model.Statistics._normalize_synergy_scores', wraps=_normalize_synergy_scores):
            mock_run_simulations.return_value = None
            compare_two_simulations(evolution_result1, evolution_result2, observed_synergy_scores, model_outputs,
                                    perturbations)
            mock_run_simulations.assert_called()
            mock_plot.assert_called()
