import os
import re
import pytest
from unittest.mock import Mock, patch, call, mock_open, MagicMock
from pydruglogics.model.ModelPredictions import ModelPredictions
from pydruglogics.model.BooleanModel import BooleanModel
import numpy as np
import pandas as pd
import logging


class TestModelPredictions:

    @pytest.fixture
    def mock_boolean_model(self):
        model = Mock(spec=BooleanModel)
        model.model_name = 'mock_model'
        model.clone.return_value = model
        model.calculate_attractors.return_value = None
        model.calculate_global_output.return_value = 0.8
        model.global_output = 0.8
        return model


    @pytest.fixture
    def mock_model_predictions(self, mock_boolean_model):
        return ModelPredictions(boolean_models=[mock_boolean_model], perturbations=[{'name': 'drug1'}],
                                model_outputs=Mock(), synergy_method='hsa')

    @pytest.fixture
    def mock_model_predictions2(self):
        mock_model = Mock(spec=BooleanModel)
        mock_model.model_name = 'mock_model'
        mock_model.calculate_global_output.return_value = 0.8
        mock_model.global_output = 0.8
        mock_model.clone.return_value = mock_model
        return ModelPredictions(boolean_models=[mock_model], perturbations=[{'name': 'drug1'}], model_outputs=Mock(),
                                synergy_method='hsa')

    def test_init_with_no_models_raises_error(self):
        with pytest.raises(ValueError, match='Please provide Boolean Models from file or list.'):
            ModelPredictions(model_directory=None)

    def test_simulate_model_responses(self, mock_model_predictions, mock_boolean_model):
        perturbation = [{'name': 'drug1'}]
        with patch('logging.debug') as mock_logging_debug:
            model, response, perturbation = mock_model_predictions._simulate_model_responses(mock_boolean_model,
                                                                                             perturbation)
            assert response == 0.8
            assert mock_logging_debug.call_count == 2

    def test_store_result_in_matrix(self, mock_model_predictions):
        output_matrix = {}
        mock_model_predictions._store_result_in_matrix(output_matrix, 'mock_model', [{'name': 'drug1'}], 0.8)
        assert 'drug1' in output_matrix
        assert 'mock_model' in output_matrix['drug1']
        assert output_matrix['drug1']['mock_model'] == 0.8

    def test_get_perturbation_name(self, mock_model_predictions):
        perturbation = [{'name': 'drug1'}, {'name': 'drug2'}]
        result = mock_model_predictions._get_perturbation_name(perturbation)
        assert result == 'drug1-drug2'

    def test_calculate_mean_responses(self, mock_model_predictions):
        mock_model_predictions._prediction_matrix = {
            'drug1': {'model1': 0.5, 'model2': 0.7},
            'drug2': {'model1': 0.8, 'model2': 0.6},
            'drug1-drug2': {'model1': 0.9, 'model2': 0.85}
        }
        mean_responses = mock_model_predictions._calculate_mean_responses()
        assert mean_responses['drug1'] == 0.6
        assert mean_responses['drug2'] == 0.7
        assert mean_responses['drug1-drug2'] == 0.875

    def test_calculate_hsa_synergy(self, mock_model_predictions):
        with patch('logging.info') as mock_logging_info:
            mock_model_predictions._calculate_hsa_synergy(0.7, 0.8, 0.6, 'drug1-drug2')

    def test_init_load_models_from_directory(self):
        with patch.object(ModelPredictions, '_load_models_from_directory') as mock_load_models:
            model_predictions = ModelPredictions(model_directory='/test/models', attractor_tool='mpbn',
                                                 attractor_type='stable_states')
            mock_load_models.assert_called_once_with(directory='/test/models', attractor_tool='mpbn',
                                                     attractor_type='stable_states')

    def test_calculate_synergy_logging_and_calculation(self, mock_model_predictions):
        mock_model_predictions._perturbations = Mock()
        mock_model_predictions._perturbations.perturbations = [[{'name': 'drug1'}, {'name': 'drug2'}],
                                                               [{'name': 'drug3'}]]

        mock_model_predictions._prediction_matrix = {'drug1': {'model1': 0.6}, 'drug2': {'model1': 0.7},
                                                     'drug1-drug2': {'model1': 0.5}, 'drug3': {'model1': 0.8}}

        mock_model_predictions._calculate_mean_responses = Mock(return_value={'drug1': 0.6, 'drug2': 0.7,
                                                                              'drug1-drug2': 0.5, 'drug3': 0.8})

        with patch('logging.info') as mock_logging_info, patch('logging.debug') as mock_logging_debug, \
                patch.object(mock_model_predictions, '_calculate_hsa_synergy') as mock_hsa, \
                patch.object(mock_model_predictions, '_calculate_bliss_synergy') as mock_bliss:

            mock_model_predictions._synergy_method = 'hsa'
            mock_model_predictions._calculate_synergy()
            mock_logging_debug.assert_called_once_with('\nCalculating synergies..')
            mock_logging_info.assert_called_with("\nSynergy scores (hsa):")
            mock_hsa.assert_called_once_with(0.5, 0.6, 0.7, 'drug1-drug2')

            mock_model_predictions._synergy_method = 'bliss'
            mock_model_predictions._calculate_synergy()
            mock_bliss.assert_called_once_with(0.5, 0.6, 0.7, 'drug1-drug2')

    def test_calculate_hsa_synergy_less_than_min_single_drug_response(self, mock_model_predictions):
        with patch('logging.info') as mock_logging_info:
            mock_model_predictions._calculate_hsa_synergy(0.4, 0.6, 0.5, 'drug1-drug2')

            assert len(mock_model_predictions._predicted_synergy_scores) == 1
            synergy_score = mock_model_predictions._predicted_synergy_scores[0][1]
            tolerance = 1e-8
            expected_score = -0.1

            assert abs(synergy_score - expected_score) < tolerance, f"Expected {expected_score}, but got {synergy_score}"
            mock_logging_info.assert_called_once_with(f'drug1-drug2: {synergy_score}')

    def test_predicted_synergy_scores_property(self, mock_model_predictions):
        mock_model_predictions._predicted_synergy_scores = [('drug1-drug2', 0.2), ('drug3-drug4', -0.1)]
        assert mock_model_predictions.predicted_synergy_scores == [('drug1-drug2', 0.2), ('drug3-drug4', -0.1)]

    def test_save_to_file_predictions_success(self, mock_model_predictions):
        mock_model_predictions._boolean_models = [Mock(model_name='model1', global_output=0.75)]
        mock_model_predictions._prediction_matrix = {'drug1': {'model1': 0.8},'drug1-drug2': {'model1': 0.6}}
        mock_model_predictions._predicted_synergy_scores = [('drug1-drug2', 0.05)]

        with patch('os.makedirs'), patch('builtins.open', mock_open()) as mock_file:
            mock_model_predictions.save_to_file_predictions(base_folder='/test/predictions')
            file_open_calls = [call[0][0] for call in mock_file.call_args_list]
            assert any(
                re.match(r'/test/predictions/predictions_\d{4}_\d{2}_\d{2}_\d{4}/model_scores\.tab', path) for
                path in file_open_calls)

            assert any(
                re.match(r'/test/predictions/predictions_\d{4}_\d{2}_\d{2}_\d{4}/synergies_hsa\.tab', path) for
                path in file_open_calls)

    def test_save_to_file_predictions_io_error(self, mock_model_predictions):
        with patch('os.makedirs'), patch('builtins.open', side_effect=IOError("Error writing file")):
            with pytest.raises(IOError, match="Error writing file"):
                mock_model_predictions.save_to_file_predictions(base_folder='/test/predictions')

    def test_get_prediction_matrix(self, mock_model_predictions):
        # Arrange: Set up the mock data for the prediction matrix
        mock_model_predictions._prediction_matrix = {
            'drug1-drug2': {'model1': 0.6},
            'drug1': {'model1': 0.8},
            'drug3': {'model1': 0.9}
        }

        # Mock the pandas DataFrame and the logging
        with patch('pandas.DataFrame.from_dict', return_value=Mock()) as mock_from_dict, \
                patch('logging.debug') as mock_logging_debug:
            # Act: Call the method to test
            mock_model_predictions.get_prediction_matrix()

            # Assert: Check that the DataFrame creation was called with the correct filtered data
            mock_from_dict.assert_called_once_with({'drug1-drug2': {'model1': 0.6}}, orient='index')

            # Assert: Check that the appropriate logging calls were made
            mock_logging_debug.assert_any_call("\nResponse Matrix:")
            mock_logging_debug.assert_any_call(mock_from_dict.return_value.fillna())

    def test_run_simulations_serial(self, mock_model_predictions):
        mock_model_predictions._boolean_models = [Mock(model_name='model1')]
        mock_model_predictions._perturbations = Mock()
        mock_model_predictions._perturbations.perturbations = [[{'name': 'drug1'}]]

        with patch.object(mock_model_predictions, '_simulate_model_responses',
                          return_value=(Mock(), 0.8, [{'name': 'drug1'}])) as mock_simulate, \
                patch.object(mock_model_predictions, '_store_result_in_matrix') as mock_store:
            mock_model_predictions.run_simulations(parallel=False)
            mock_simulate.assert_called_once_with(mock_model_predictions._boolean_models[0], [{'name': 'drug1'}])
            mock_store.assert_called_once()

    def test_calculate_bliss_synergy(self, mock_model_predictions):
        mock_model_predictions._model_outputs = Mock(min_output=0, max_output=1)

        with patch('logging.info') as mock_logging_info:
            mock_model_predictions._calculate_bliss_synergy(0.5, 0.7, 0.6, 'drug1-drug2')

            assert len(mock_model_predictions._predicted_synergy_scores) == 1
            perturbation, synergy_score = mock_model_predictions._predicted_synergy_scores[0]
            assert perturbation == 'drug1-drug2'
            assert synergy_score == pytest.approx(0.08)

    def test_calculate_hsa_synergy_score_zero(self, mock_model_predictions):
        with patch('logging.info') as mock_logging_info:
            mock_model_predictions._calculate_hsa_synergy(0.7, 0.7, 0.7, 'drug1-drug2')

            assert len(mock_model_predictions._predicted_synergy_scores) == 1
            perturbation, synergy_score = mock_model_predictions._predicted_synergy_scores[0]
            assert perturbation == 'drug1-drug2'
            assert synergy_score == 0

            mock_logging_info.assert_called_once_with(f'drug1-drug2: {synergy_score}')

    def test_run_simulations_parallel_execution(self, mock_model_predictions2):
        mock_model_predictions2._boolean_models = [Mock(model_name='model1'), Mock(model_name='model2')]
        mock_model_predictions2._perturbations = Mock()
        mock_model_predictions2._perturbations.perturbations = [[{'name': 'drug1'}], [{'name': 'drug2'}]]

        with patch.object(mock_model_predictions2, '_simulate_model_responses', side_effect=[
            (Mock(model_name='model1'), 0.8, [{'name': 'drug1'}]),
            (Mock(model_name='model2'), 0.7, [{'name': 'drug2'}])
        ]) as mock_simulate, patch.object(mock_model_predictions2, '_store_result_in_matrix') as mock_store:
            mock_model_predictions2.run_simulations(parallel=True)

            assert mock_store.call_count == 4

    def test_save_to_file_predictions_general_exception(self, mock_model_predictions2):
        with patch('os.makedirs'), patch('builtins.open', side_effect=Exception("Unexpected error")), \
                patch('logging.error') as mock_logging_error:
            with pytest.raises(Exception, match="Unexpected error"):
                mock_model_predictions2.save_to_file_predictions(base_folder='/test/predictions')

            mock_logging_error.assert_called_once_with("Error saving predictions to file: Unexpected error")

    def test_load_models_from_directory_success_single_file(self):
        mock_bnet_data = """A, B & C
                            B, !C | D
                            C, A"""

        with patch('os.listdir', return_value=['model1.bnet']), \
                patch('pydruglogics.model.BooleanModel') as mock_boolean_model, \
                patch('os.path.join', side_effect=lambda *args: '/'.join(args)), \
                patch('builtins.open', mock_open(read_data=mock_bnet_data)):

            mock_boolean_model_instance = mock_boolean_model.return_value
            mock_boolean_model_instance.calculate_attractors = MagicMock()
            mock_boolean_model_instance.calculate_global_output = MagicMock()
            mock_model_outputs = Mock()
            mock_model_outputs.model_outputs = {'A': 0.5,'B': 1.0, 'C': 0.75}
            model_predictions = ModelPredictions(model_outputs=mock_model_outputs, model_directory='mock_directory',
                                                 boolean_models=None, attractor_tool='pyboolnet',
                                                 attractor_type='stable_states')
            model_predictions._load_models_from_directory('mock_directory', 'pyboolnet',
                                                          'stable_states')
            assert len(model_predictions._boolean_models) == 2

    @pytest.fixture
    def mock_model_outputs(self):
        mock_outputs = MagicMock()
        mock_outputs.model_outputs = {
            'A': 0.5,
            'B': 1.0,
            'C': 0.75
        }
        mock_outputs.min_output = 0.5
        mock_outputs.max_output = 1
        return mock_outputs

    def test_general_error_loading_models(self, mock_model_outputs):
        with patch('os.listdir', side_effect=Exception("Permission denied")), \
                patch('logging.error') as mock_logging_error:
            with pytest.raises(Exception) as exc_info:
                ModelPredictions(model_outputs=mock_model_outputs, model_directory='mock_directory',
                                 attractor_tool='pyboolnet',attractor_type='stable_states')

            mock_logging_error.assert_called_with("Error loading models from directory mock_"
                                                  "directory: Permission denied")
            assert "Permission denied" in str(exc_info.value)

    def test_error_loading_specific_model_file(self, mock_model_outputs, capsys):
        invalid_bnet_data = "invalid content"

        with patch('os.listdir', return_value=['broken_model.bnet']), \
             patch('builtins.open', mock_open(read_data=invalid_bnet_data)), \
             patch('os.path.join', return_value='mock_directory/broken_model.bnet'), \
             patch('logging.error') as mock_logging_error:

            with pytest.raises(ValueError, match="not enough values to unpack"):
                ModelPredictions(model_outputs=mock_model_outputs, model_directory='mock_directory',
                                 attractor_tool='pyboolnet', attractor_type='stable_states')

            captured = capsys.readouterr()
            assert ("Failed to load model from mock_directory/broken_model.bnet: not enough "
                    "values to unpack") in captured.out
            mock_logging_error.assert_called_with("Error loading models from directory mock_directory: "
                                                  "not enough values to unpack (expected 2, got 1)")
