import pytest
from unittest.mock import patch, MagicMock
from pydruglogics.utils.BNetworkUtil import BNetworkUtil
import logging
from pydruglogics.input.ModelOutputs import ModelOutputs


class TestModelOutputs:
    @pytest.fixture
    def model_outputs(self):
        return ModelOutputs(input_dictionary={"A": 1.0, "B": -1.0, "C": 1.0})

    def test_init_with_file(self):
        with patch.object(ModelOutputs, '_load_model_outputs_file') as mock_load_file:
            model_outputs = ModelOutputs(input_file='dummy_file')
            mock_load_file.assert_called_once_with('dummy_file')

    def test_init_with_dict(self):
        model_dict = {"A": 1.0, "B": -1.0}
        instance = ModelOutputs(input_dictionary=model_dict)
        assert instance.model_outputs == model_dict

    def test_init_with_no_input(self):
        with pytest.raises(ValueError):
            ModelOutputs()

    def test_size(self, model_outputs):
        assert model_outputs.size() == 3

    def test_get_model_output_existing(self, model_outputs):
        assert model_outputs.get_model_output("A") == 1.0

    def test_get_model_output_non_existing(self, model_outputs):
        assert model_outputs.get_model_output("Z") == 0.0

    def test_calculate_max_output(self, model_outputs):
        assert model_outputs.max_output == 2.0  # 1.0 + 3.0

    def test_calculate_min_output(self, model_outputs):
        assert model_outputs.min_output == -1.0  # Only negative weight

    def test_node_names(self, model_outputs):
        assert set(model_outputs.node_names) == {"A", "B", "C"}

    def test_print_method(self, model_outputs):
        with patch('builtins.print') as mock_print:
            model_outputs.print()
            mock_print.assert_called()

    def test_str_method(self, model_outputs):
        output_string = str(model_outputs)
        expected_string = "Model output: A, weight: 1.0\nModel output: B, weight: -1.0\nModel output: C, weight: 1.0"
        assert output_string == expected_string

    def test_edge_case_large_numbers(self):
        large_model_dict = {"A": 1e6, "B": -1e6}
        instance = ModelOutputs(input_dictionary=large_model_dict)
        assert instance.max_output == 1e6
        assert instance.min_output == -1e6

    def test_edge_case_identical_weights(self):
        identical_weights_dict = {"A": 0.0, "B": 0.0, "C": 0.0}
        instance = ModelOutputs(input_dictionary=identical_weights_dict)
        assert instance.max_output == 0.0
        assert instance.min_output == 0.0

    def test_init_with_dict(self):
        model_outputs_dict = {"A": 1.0}
        model_outputs = ModelOutputs(input_dictionary=model_outputs_dict)
        assert model_outputs.size() == 1
        assert model_outputs.max_output == 1.0
        assert model_outputs.min_output == 0.0

    def test_init_with_large_file(self):
        large_input = [f"A{i}\t{i}" for i in range(1000)]
        with patch.object(BNetworkUtil, 'read_lines_from_file', return_value=large_input):
            model_outputs = ModelOutputs(input_file='dummy_file')
            assert model_outputs.size() == 1000
            assert model_outputs.max_output == sum(range(1000))
            assert model_outputs.min_output == 0.0

    def test_load_model_outputs_file_io_error(self):
        with patch.object(BNetworkUtil, 'read_lines_from_file', side_effect=IOError("File read error")):
            with pytest.raises(IOError, match="File read error"):
                ModelOutputs(input_file='dummy_file')

    def test_load_model_outputs_file_general_exception(self):
        with patch.object(BNetworkUtil, 'read_lines_from_file', side_effect=Exception("Unexpected error")):
            with pytest.raises(Exception, match="Unexpected error"):
                ModelOutputs(input_file='dummy_file')

    def test_load_model_outputs_dict_exception(self):
        with patch.object(ModelOutputs, '_load_model_outputs_dict', side_effect=Exception("Dictionary error")):
            with pytest.raises(Exception, match="Dictionary error"):
                ModelOutputs(input_dictionary={"A": 1.0})

    def test_print_exception(self, model_outputs):
        with patch('builtins.print', side_effect=Exception("Print error")):
            with pytest.raises(Exception, match="Print error"):
                model_outputs.print()

    def test_load_model_outputs_dict_valid(self):
        instance = ModelOutputs(input_dictionary={"A": 1.0, "B": -2.0})
        assert instance.model_outputs == {"A": 1.0, "B": -2.0}

    def test_load_model_outputs_dict_empty_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            instance = ModelOutputs(input_dictionary={})
        assert "model_outputs_dict is empty. No model outputs loaded." in caplog.text
        assert instance.model_outputs == {}

    def test_load_model_outputs_dict_invalid_type(self):
        with pytest.raises(TypeError, match="model_outputs_dict must be a dictionary."):
            ModelOutputs(input_dictionary=["Not", "a", "dictionary"])
