import pytest
from unittest.mock import patch
from gitsbe.input.TrainingData import TrainingData
from gitsbe.utils.Util import Util


class TestTrainingData:

    @pytest.fixture(autouse=True)
    def reset(self):
        TrainingData.training_data = None
        yield
        TrainingData.training_data = None

    @pytest.fixture
    def mock_file_content(self):
        return [
            "condition",
            "cond1\tcond2",
            "response",
            "globaloutput:0.5",
            "weight:1.0",
            "condition",
            "cond3\tcond4",
            "response",
            "A:1",
            "weight:2.0"
        ]

    def test_singleton_initialization(self, mock_file_content):
        with patch.object(Util, 'read_lines_from_file', return_value=mock_file_content):
            TrainingData.initialize('dummy_file')
            with pytest.raises(AssertionError, match='The TrainingData class has already been initialized'):
                TrainingData.initialize('dummy_file')

    def test_initialize_and_get_instance(self, mock_file_content):
        with patch.object(Util, 'read_lines_from_file', return_value=mock_file_content):
            TrainingData.initialize('dummy_file')
            instance = TrainingData.get_instance()
            assert isinstance(instance, TrainingData)
            assert instance.size() == 2

    def test_get_instance_without_initialization(self):
        with pytest.raises(AssertionError, match='You have to call init first to initialize the TrainingData class'):
            TrainingData.get_instance()

    def test_load_from_file(self, mock_file_content):
        with patch.object(Util, 'read_lines_from_file', return_value=mock_file_content):
            data = TrainingData('dummy_file')
            assert data.size() == 2
            assert data.get_weight_sum == 3.0
