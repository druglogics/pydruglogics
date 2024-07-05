import pytest
from unittest.mock import Mock, patch, mock_open
from gitsbe import BooleanModel
from gitsbe.input.ModelOutputs import ModelOutputs
from gitsbe.utils.Util import Util


class TestBooleanModel:

    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.model_name = 'test_model'
        model.interactions = [{'source': 'A', 'target': 'B'}, {'source': 'B', 'target': 'C'}]
        return model

    @pytest.fixture
    def boolean_model(self):
        return BooleanModel()

    def test_init_from_file_invalid_extension(self, boolean_model):
        with pytest.raises(IOError):
            with patch('builtins.open', mock_open(read_data="data")):
                boolean_model.init_from_file('model.txt', 'biolqm_trapspaces')

    def test_init_from_file(self, boolean_model):
        with patch('builtins.open', mock_open(read_data="A, B, 0\nB, C, 1")), \
                patch.object(Util, 'get_file_extension', return_value='bnet'):
            boolean_model.init_from_file('model.bnet', 'biolqm_trapspaces')
            assert boolean_model._model_name == 'model'
            assert len(boolean_model._boolean_equations) == 2

    def test_calculate_global_output(self, boolean_model):
        mock_outputs = Mock()
        mock_outputs.get_instance.return_value = mock_outputs
        mock_outputs.model_outputs = [{'node_name': 'A', 'weight': 1}, {'node_name': 'B', 'weight': 2}]
        mock_outputs.min_output = 0
        mock_outputs.max_output = 10

        boolean_model._attractors = [{'A': 1, 'B': 0}, {'A': 0, 'B': 1}]

        with patch.object(ModelOutputs, 'get_instance', return_value=mock_outputs):
            result = boolean_model.calculate_global_output()
            assert result == 0.15  # (0.5 + 1)/2 / 10

    def test_reset_attractors(self, boolean_model):
        boolean_model._attractors = ['state1', 'state2']
        boolean_model.reset_attractors()
        assert not boolean_model._attractors

    def test_has_attractors(self, boolean_model):
        boolean_model._attractors = ['state1']
        assert boolean_model.has_attractors() is True
        boolean_model._attractors = []
        assert boolean_model.has_attractors() is False

    def test_boolean_equations_property(self, boolean_model):
        boolean_model._boolean_equations = ['equation1', 'equation2']
        assert boolean_model.boolean_equations == ['equation1', 'equation2']

    def test_attractors_property(self, boolean_model):
        boolean_model._attractors = ['state1', 'state2']
        assert boolean_model.attractors == ['state1', 'state2']

    def test_attractor_tool_property(self, boolean_model):
        boolean_model._attractor_tool = 'biolqm'
        assert boolean_model.attractor_tool == 'biolqm'

    def test_str(self, boolean_model):
        boolean_model._attractors = ['state1']
        boolean_model._boolean_equations = ['equation1']
        assert str(boolean_model) == "Attractors: ['state1']"

    def test_update_boolean_model_balance(self, boolean_model):
        boolean_model._boolean_equations = [Mock(), Mock()]
        solution = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
        boolean_model.update_boolean_model_balance(solution)
        for eq in boolean_model._boolean_equations:
            eq.modify_link_from_list.assert_called()

    def test_update_boolean_model_both(self, boolean_model):
        boolean_model._boolean_equations = [Mock(), Mock()]
        solution = [0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1]
        boolean_model.update_boolean_model_both(solution)
        for eq in boolean_model._boolean_equations:
            eq.modify_activating_values_from_list.assert_called()
            eq.modify_inhibitory_values_from_list.assert_called()
            eq.modify_link_from_list.assert_called()

    def test_update_boolean_model_topology(self, boolean_model):
        boolean_model._boolean_equations = [Mock(), Mock()]
        solution = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
        boolean_model.update_boolean_model_topology(solution)
        for eq in boolean_model._boolean_equations:
            eq.modify_activating_values_from_list.assert_called()
            eq.modify_inhibitory_values_from_list.assert_called()
