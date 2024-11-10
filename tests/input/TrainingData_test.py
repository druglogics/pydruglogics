import pytest
from unittest.mock import patch, MagicMock
from pydruglogics.input.TrainingData import TrainingData
from pydruglogics.utils.BNetworkUtil import BNetworkUtil
import logging

class TestTrainingData:
    @pytest.fixture
    def training_data_globaloutput(self):
        observations = [(['globaloutput:0.1'], 0.5)]
        return TrainingData(observations=observations)

    @pytest.fixture
    def training_data_multiple_response(self):
        return TrainingData(observations=[(['A:0', 'B:1', 'C:0', 'D:0.453'], 1.0)])

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    def test_init_without_input(self):
        with pytest.raises(ValueError):
            TrainingData()

    def test_init_with_file_globaloutput(self):
        with patch.object(TrainingData, '_load_from_file') as mock_load_file:
            TrainingData(input_file='dummy_file')
            mock_load_file.assert_called_once_with('dummy_file')

    def test_load_from_observations_list(self):
        observations = [(['Response1:1', 'Response2:1'], 1.0)]
        training_data = TrainingData(observations=observations)
        assert training_data.size() == 1
        assert training_data.observations[0]['condition'] == ['-']
        assert training_data.observations[0]['response'] == ['Response1:1', 'Response2:1']
        assert training_data.observations[0]['weight'] == 1.0

    def test_load_from_observations_with_globaloutput(self):
        observations = [(['globaloutput:0.5'], 1.0),]
        training_data = TrainingData(observations=observations)
        assert training_data.size() == 1
        assert training_data.observations[0]['response'] == ['globaloutput:0.5']

    def test_load_from_file_with_valid_globaloutput(self):
        with patch.object(BNetworkUtil, 'read_lines_from_file', return_value=[
            "condition",
            "A:0",
            "response",
            "globaloutput:0.5",
            "weight: 1.0"
        ]):
            training_data = TrainingData(input_file='dummy_file')
            assert training_data.size() == 1
            assert training_data.observations[0]['response'] == ['globaloutput:0.5']

    def test_load_from_file_with_invalid_globaloutput_non_numeric(self):
        with patch.object(BNetworkUtil, 'read_lines_from_file', return_value=[
            "condition",
            "A:0",
            "response",
            "globaloutput:asd",
            "weight: 1.0"
        ]):
            with pytest.raises(ValueError) as exc_info:
                TrainingData(input_file='dummy_file')
            assert exc_info.value.args[0] == "Response: ['globaloutput:asd'] has a non-numeric value: asd"

    def test_load_from_file_with_invalid_globaloutput_out_of_range(self):
        with patch.object(BNetworkUtil, 'read_lines_from_file', return_value=[
            "condition\n",
            "A:0\n",
            "response\n",
            "globaloutput:1.5\n",
            "weight: 1.0\n"
        ]):
            with pytest.raises(ValueError) as exc_info:
                TrainingData(input_file='dummy_file')
            assert exc_info.value.args[0] == 'Response has globaloutput outside the [-1,1] range: 1.5\n'

    def test_add_observation_with_invalid_globaloutput_non_numeric(self):
        training_data = TrainingData(observations=[])
        with pytest.raises(ValueError) as exc_info:
            training_data._add_observation(['-'], ['globaloutput:abc'], 0.2)
        assert str(exc_info.value) == "Response: ['globaloutput:abc'] has a non-numeric value: abc"

    def test_add_observation_with_invalid_globaloutput_out_of_range(self):
        training_data = TrainingData(observations=[])
        with pytest.raises(ValueError) as exc_info:
            training_data._add_observation(['-'], ['globaloutput:2.0'], 0.2)
        assert str(exc_info.value) == "Response has globaloutput outside the [-1,1] range: 2.0"

    def test_print_method(self, training_data_globaloutput):
        with patch('builtins.print') as mock_print:
            training_data_globaloutput.print()
            mock_print.assert_called()

    def test_no_training_data_available(self):
        observations = []
        training_data = TrainingData(observations=observations)
        assert str(training_data) == 'No observations available.'

    def test_no_training_data_available(self):
        observations = []
        training_data = TrainingData(observations=observations)
        assert str(training_data) == 'No observations available.'

    def test_weights_property(self, training_data_multiple_response):
        assert training_data_multiple_response.weights == [1]

    def test_response_property(self, training_data_multiple_response):
        assert training_data_multiple_response.response == ['A:0', 'B:1', 'C:0', 'D:0.453']

    def test_responses_property(self, training_data_multiple_response):
        assert training_data_multiple_response.responses == ['A:0', 'B:1', 'C:0', 'D:0.453']

    def test_weight_sum_property(self, training_data_multiple_response):
        assert training_data_multiple_response.weight_sum == 1.0

    def test_print_method_exception_handling(self, training_data_globaloutput):
        with patch.object(TrainingData, '__str__', side_effect=RuntimeError("Test error")), \
                patch('builtins.print') as mock_print, \
                patch('logging.error') as mock_log_error:
            with pytest.raises(RuntimeError, match="Test error"):
                training_data_globaloutput.print()

            mock_log_error.assert_called_once_with("Error while printing TrainingData: Test error")

    def test_missing_condition_or_response(self):
        with patch.object(BNetworkUtil, 'read_lines_from_file', return_value=[
            "weight: 1.0\n"
        ]):
            with pytest.raises(ValueError, match="Missing condition or response data before the weight entry."):
                TrainingData(input_file='dummy_file')

    def test_load_from_file_io_error_handling(self):
        with patch.object(BNetworkUtil, 'read_lines_from_file', side_effect=IOError("File not found")), \
                patch('logging.error') as mock_log_error:
            with pytest.raises(IOError, match="File not found"):
                TrainingData(input_file='dummy_file')

            mock_log_error.assert_called_once_with("File read error: File not found")
