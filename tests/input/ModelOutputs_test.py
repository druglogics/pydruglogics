import pytest
from unittest.mock import patch, mock_open
from gitsbe.utils.Util import Util
from gitsbe.input.ModelOutputs import ModelOutputs


class TestModelOutputs:

    @pytest.fixture(autouse=True)
    def reset(self):
        ModelOutputs.reset()
        yield
        ModelOutputs.reset()

    def test_initialize(self):
        with patch.object(Util, 'read_lines_from_file', return_value=["A\t1.0", "B\t2.0"]):
            ModelOutputs.initialize('dummy_file')
            instance = ModelOutputs.get_instance()
            assert instance.size() == 2

    def test_get_instance_without_initialization(self):
        with pytest.raises(AssertionError, match='You have to call init first to initialize the ModelOutputs'):
            ModelOutputs.get_instance()

    def test_reset(self):
        with patch.object(Util, 'read_lines_from_file', return_value=["A\t1.0", "B\t2.0"]):
            ModelOutputs.initialize('dummy_file')
            ModelOutputs.reset()
            with pytest.raises(AssertionError, match='You have to call init first to initialize the ModelOutputs'):
                ModelOutputs.get_instance()

    def test_get_model_output(self):
        with patch.object(Util, 'read_lines_from_file', return_value=["A\t1.0", "B\t2.0"]):
            ModelOutputs.initialize('dummy_file')
            instance = ModelOutputs.get_instance()
            assert instance.get_model_output(0) == {'node_name': 'A', 'weight': 1.0}
            assert instance.get_model_output(1) == {'node_name': 'B', 'weight': 2.0}

    def test_calculate_max_output(self):
        with patch.object(Util, 'read_lines_from_file', return_value=["A\t1.0", "B\t-2.0", "C\t3.0"]):
            ModelOutputs.initialize('dummy_file')
            instance = ModelOutputs.get_instance()
            assert instance.max_output == 4.0

    def test_calculate_min_output(self):
        with patch.object(Util, 'read_lines_from_file', return_value=["A\t1.0", "B\t-2.0", "C\t-3.0"]):
            ModelOutputs.initialize('dummy_file')
            instance = ModelOutputs.get_instance()
            assert instance.min_output == -5.0
