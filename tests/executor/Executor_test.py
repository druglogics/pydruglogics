import pytest
import time
import logging
from unittest.mock import patch, Mock, call
from pydruglogics.model.BooleanModel import BooleanModel
from pydruglogics.input.ModelOutputs import ModelOutputs
from pydruglogics.input.TrainingData import TrainingData
from pydruglogics.input.Perturbations import Perturbation
from pydruglogics.model.Evolution import Evolution
from pydruglogics.execution.Executor import train, predict, execute, timed_execution

class TestExecutor:

    @pytest.fixture
    def boolean_model(self):
        return Mock(spec=BooleanModel)

    @pytest.fixture
    def model_outputs(self):
        return Mock(spec=ModelOutputs)

    @pytest.fixture
    def training_data(self):
        return Mock(spec=TrainingData)

    @pytest.fixture
    def perturbations(self):
        return Mock(spec=Perturbation)

    @pytest.fixture
    def evolution(self, boolean_model, model_outputs, training_data):
        return Mock(spec=Evolution)

    @patch('pydruglogics.execution.Executor.time.time', return_value=1000)
    @patch('pydruglogics.execution.Executor.logging.info')
    def test_timed_execution(self, mock_logging_info, mock_time):
        @timed_execution
        def sample_function():
            time.sleep(0.1)
            return "Done"

        result = sample_function()
        assert result == "Done"
        mock_logging_info.assert_any_call('Sample_function started...')
        mock_logging_info.assert_any_call('Sample_function completed in 0.00 seconds.')



    @patch('pydruglogics.execution.Executor.train')
    @patch('pydruglogics.execution.Executor.predict')
    @patch('pydruglogics.execution.Executor.logging.info')
    @patch('pydruglogics.execution.Executor.time.time', side_effect=[1000, 1005])
    def test_execute(self, mock_time, mock_logging_info, mock_predict, mock_train):
        train_params = {'boolean_model': Mock(), 'model_outputs': Mock(), 'ga_args': {}, 'ev_args': {}}
        predict_params = {'model_outputs': Mock(), 'perturbations': Mock(), 'observed_synergy_scores': ['SC1']}
        mock_train.return_value = [Mock(spec=BooleanModel)]

        execute(train_params, predict_params)

        mock_train.assert_called_once_with(**train_params)
        mock_predict.assert_called_once_with(best_boolean_models=mock_train.return_value, **predict_params)
        mock_logging_info.assert_any_call('Total runtime for training and prediction: 5.00 seconds')

    @patch('pydruglogics.execution.Executor.predict')
    def test_execute_with_predict_params_and_without_train_params(self, mock_predict):
        predict_params = {
            'model_outputs': Mock(spec=ModelOutputs),
            'perturbations': Mock(spec=Perturbation),
            'observed_synergy_scores': ['SC1']
        }

        execute(train_params=None, predict_params=predict_params)

        mock_predict.assert_called_once_with(best_boolean_models=None, **predict_params)

    @patch('pydruglogics.execution.Executor.logging.info')
    @patch('pydruglogics.execution.Executor.time.time', side_effect=[1000, 1020])
    def test_execute_logging_total_runtime(self, mock_time, mock_logging_info):
        train_params = {
            'boolean_model': Mock(),
            'model_outputs': Mock(),
            'ga_args': {},
            'ev_args': {}
        }
        predict_params = {
            'model_outputs': Mock(),
            'perturbations': Mock(),
            'observed_synergy_scores': ['SC1']
        }

        with (patch('pydruglogics.execution.Executor.train',return_value=[Mock(spec=BooleanModel)])
              as mock_train, patch('pydruglogics.execution.Executor.predict') as mock_predict):
            execute(train_params, predict_params)

            mock_logging_info.assert_any_call('Total runtime for training and prediction: 20.00 seconds')

    @patch('pydruglogics.execution.Executor.ModelPredictions')
    @patch('pydruglogics.execution.Executor.PlotUtil.plot_roc_and_pr_curve')
    @patch('pydruglogics.execution.Executor.logging.info')
    def test_predict_with_models_form_directory(self, mock_logging_info, mock_plot, mock_model_predictions,
                                                model_outputs, perturbations):
        mock_model_predictions_instance = mock_model_predictions.return_value
        mock_model_predictions_instance.predicted_synergy_scores = ['synergy_score']

        predict_params = {
            'best_boolean_models': None,
            'model_outputs': model_outputs,
            'perturbations': perturbations,
            'observed_synergy_scores': ['SC1'],
            'run_parallel': True,
            'save_predictions': False,
            'save_path': './predictions',
            'model_directory': './models',
            'attractor_tool': 'mpbn',
            'attractor_type': 'stable_states',
            'synergy_method': 'hsa',
            'cores': 4
        }

        predict(**predict_params)

        mock_logging_info.assert_called_with('Predict completed in 0.00 seconds.')
        mock_model_predictions.assert_called_once_with(
            model_outputs=model_outputs,
            perturbations=perturbations,
            model_directory='./models',
            attractor_tool='mpbn',
            attractor_type='stable_states',
            synergy_method='hsa',
            boolean_models=None
        )
        mock_plot.assert_called_once_with(
            mock_model_predictions_instance.predicted_synergy_scores,
            ['SC1'],
            'hsa'
        )

    @patch('pydruglogics.execution.Executor.ModelPredictions')
    @patch('pydruglogics.execution.Executor.PlotUtil.plot_roc_and_pr_curve')
    @patch('pydruglogics.execution.Executor.logging.info')
    @patch('pydruglogics.execution.Executor.time.time', side_effect=[1000, 1025])
    def test_predict_with_existing_models(self, mock_time, mock_logging_info, mock_plot, mock_model_predictions,
                                          boolean_model, model_outputs, perturbations):
        mock_model_predictions_instance = mock_model_predictions.return_value
        mock_model_predictions_instance.predicted_synergy_scores = ['synergy_score']

        predict_params = {
            'best_boolean_models': [boolean_model],
            'model_outputs': model_outputs,
            'perturbations': perturbations,
            'observed_synergy_scores': ['SC1'],
            'run_parallel': True,
            'save_predictions': True,
            'save_path': './predictions',
            'synergy_method': 'hsa',
            'cores': 4
        }

        predict(**predict_params)

        mock_model_predictions.assert_called_once_with(
            boolean_models=[boolean_model],
            perturbations=perturbations,
            model_outputs=model_outputs,
            synergy_method='hsa'
        )
        mock_model_predictions_instance.run_simulations.assert_called_once_with(True, 4)
        mock_model_predictions_instance.save_to_file_predictions.assert_called_once_with('./predictions')
        mock_plot.assert_called_once_with(
            mock_model_predictions_instance.predicted_synergy_scores,
            ['SC1'],
            'hsa'
        )
        mock_logging_info.assert_any_call('Predict completed in 25.00 seconds.')

    @patch('pydruglogics.execution.Executor.Evolution')
    @patch('pydruglogics.execution.Executor.logging.info')
    def test_train_with_save_best_models(self, mock_logging_info, mock_evolution, boolean_model, model_outputs):
        mock_evolution_instance = mock_evolution.return_value
        mock_evolution_instance.run.return_value = [Mock(spec=BooleanModel)]

        train_params = {
            'boolean_model': boolean_model,
            'model_outputs': model_outputs,
            'training_data': Mock(),
            'ga_args': {'param1': 'value1'},
            'ev_args': {'param2': 'value2'},
            'save_best_models': True,
            'save_path': './models'
        }

        result = train(**train_params)

        mock_evolution.assert_called_once_with(
            boolean_model=boolean_model,
            model_outputs=model_outputs,
            training_data=train_params['training_data'],
            ga_args=train_params['ga_args'],
            ev_args=train_params['ev_args']
        )
        mock_evolution_instance.run.assert_called_once()
        mock_evolution_instance.save_to_file_models.assert_called_once_with('./models')
        assert result == mock_evolution_instance.run.return_value


