import pytest
from unittest.mock import patch, MagicMock
from pydruglogics.utils.BNetworkUtil import BNetworkUtil
import logging
from pydruglogics.model.InteractionModel import InteractionModel

class TestInteractionModel:
    @pytest.fixture
    def interaction_model(self):
        with patch.object(InteractionModel, '_load_sif_file'):
            return InteractionModel(interactions_file='dummy_file.sif', model_name='TestModel')

    def test_init_with_file(self):
        with patch.object(InteractionModel, '_load_sif_file') as mock_load_file:
            model = InteractionModel(interactions_file='dummy_file')
            mock_load_file.assert_called_once_with('dummy_file')

    def test_init_without_file(self):
        with pytest.raises(ValueError, match="An 'interactions_file' must be provided for initialization."):
            InteractionModel()

    def test_load_sif_file_with_invalid_extension(self):
        with patch.object(BNetworkUtil, 'get_file_extension', return_value='txt'):
            with pytest.raises(IOError, match="ERROR: The extension needs to be .sif \(other formats not yet supported\)"):
                InteractionModel(interactions_file='dummy_file.txt')

    def test_load_sif_file_io_error(self):
        with patch.object(BNetworkUtil, 'read_lines_from_file', side_effect=IOError("File not found")), \
             patch('logging.error') as mock_log_error:
            with pytest.raises(IOError, match="File not found"):
                InteractionModel(interactions_file='dummy_file.sif')
            mock_log_error.assert_called_once_with("Error reading the interactions file 'dummy_file.sif': File not found")

    def test_load_sif_file_general_exception(self):
        with patch.object(BNetworkUtil, 'read_lines_from_file', side_effect=Exception("Unexpected error")), \
             patch('logging.error') as mock_log_error:
            with pytest.raises(Exception, match="Unexpected error"):
                InteractionModel(interactions_file='dummy_file.sif')
            mock_log_error.assert_called_once_with("An unexpected error occurred while loading the file "
                                                   "'dummy_file.sif': Unexpected error")

    def test_load_sif_file_success(self):
        sif_lines = [
            "A\t->\tB",
            "C\t->\tB",
            "A\t->\tC",
            "D\t-|\tB",
            "E\t-|\tC"
        ]
        with patch.object(BNetworkUtil, 'read_lines_from_file', return_value=sif_lines), \
             patch.object(BNetworkUtil, 'parse_interaction', side_effect=lambda x: {'source': x.split()[0],
                                                                                    'target': x.split()[-1], 'arc': 1}):
            model = InteractionModel(interactions_file='dummy_file.sif')
            assert len(model.interactions) == 5

    def test_remove_self_regulated_interactions(self, interaction_model):
        interaction_model._interactions = [
            {'source': 'A', 'target': 'A', 'arc': 1},
            {'source': 'B', 'target': 'D', 'arc': 1}
        ]
        interaction_model._remove_self_regulated_interactions()
        assert len(interaction_model.interactions) == 1
        assert {'source': 'B', 'target': 'D', 'arc': 1} in interaction_model.interactions

    def test_remove_interactions_inputs(self, interaction_model):
        interaction_model._interactions = [
            {'source': 'A', 'target': 'D', 'arc': 1},
            {'source': 'D', 'target': 'A', 'arc': 1},
            {'source': 'B', 'target': 'A', 'arc': 1},
            {'source': 'B', 'target': 'A', 'arc': -1}
        ]
        interaction_model._remove_interactions(is_input=True)
        assert len(interaction_model.interactions) == 2
        assert {'source': 'A', 'target': 'D', 'arc': 1} in interaction_model.interactions
        assert {'source': 'D', 'target': 'A', 'arc': 1} in interaction_model.interactions

    def test_remove_interactions_outputs(self, interaction_model):
        interaction_model._interactions = [
            {'source': 'A', 'target': 'D', 'arc': 1},
            {'source': 'D', 'target': 'D', 'arc': 1},
            {'source': 'C', 'target': 'B', 'arc': 1},
            {'source': 'D', 'target': 'B', 'arc': -1}
        ]
        interaction_model._remove_interactions(is_output=True)
        assert len(interaction_model.interactions) == 2
        assert {'source': 'A', 'target': 'D', 'arc': 1} in interaction_model.interactions
        assert {'source': 'D', 'target': 'D', 'arc': 1} in interaction_model.interactions

    def test_build_multiple_interactions(self, interaction_model):
        interaction_model._interactions = [
            {'source': 'A', 'target': 'B', 'arc': 1},
            {'source': 'C', 'target': 'B', 'arc': -1}
        ]
        interaction_model._build_multiple_interactions()
        assert interaction_model.interactions[0]['target'] == 'B'
        assert interaction_model.interactions[0]['activating_regulators'] == ['A']
        assert interaction_model.interactions[0]['inhibitory_regulators'] == ['C']

    def test_print_method(self, interaction_model):
        with patch('builtins.print') as mock_print:
            interaction_model.print()
            mock_print.assert_called()

    def test_print_exception(self, interaction_model):
        with patch('builtins.print', side_effect=Exception("Print error")), \
             patch('logging.error') as mock_log_error:
            with pytest.raises(Exception, match="Print error"):
                interaction_model.print()
            mock_log_error.assert_called_once_with("An error occurred while printing the Interactions: Print error")

    def test_is_not_a_source(self, interaction_model):
        interaction_model._interactions = [
            {'source': 'A', 'target': 'B', 'arc': 1}
        ]
        assert interaction_model._is_not_a_source('B') is True
        assert interaction_model._is_not_a_source('A') is False

    def test_is_not_a_target(self, interaction_model):
        interaction_model._interactions = [
            {'source': 'A', 'target': 'B', 'arc': 1}
        ]
        assert interaction_model._is_not_a_target('A') is True
        assert interaction_model._is_not_a_target('B') is False

    def test_model_name_property(self, interaction_model):
        assert interaction_model.model_name == 'TestModel'

    def test_get_interaction(self, interaction_model):
        interaction_model._interactions = [{'source': 'A', 'target': 'B', 'arc': 1}]
        interaction = interaction_model.get_interaction(0)
        assert interaction == {'source': 'A', 'target': 'B', 'arc': 1}

    def test_get_target(self, interaction_model):
        interaction_model._interactions = [{'source': 'A', 'target': 'B', 'arc': 1}]
        target = interaction_model.get_target(0)
        assert target == 'B'

    def test_get_activating_regulators(self, interaction_model):
        interaction_model._interactions = [{'source': 'A', 'target': 'B', 'arc': 1, 'activating_regulators': ['A']}]
        regulators = interaction_model.get_activating_regulators(0)
        assert regulators == ['A']

    def test_get_inhibitory_regulators(self, interaction_model):
        interaction_model._interactions = [{'source': 'A', 'target': 'B', 'arc': -1, 'inhibitory_regulators': ['A']}]
        regulators = interaction_model.get_inhibitory_regulators(0)
        assert regulators == ['A']

    def test_all_targets_property(self, interaction_model):
        interaction_model._interactions = [
            {'source': 'A', 'target': 'B', 'arc': 1},
            {'source': 'C', 'target': 'D', 'arc': -1}
        ]
        assert set(interaction_model.all_targets) == {'B', 'D'}

    def test_str_method(self, interaction_model):
        interaction_model._interactions = [
            {'target': 'B', 'activating_regulators': ['A'], 'inhibitory_regulators': []},
            {'target': 'D', 'activating_regulators': [], 'inhibitory_regulators': ['C']}
        ]
        output_string = str(interaction_model)
        expected_string = (
            'Target: B, activating regulators: A\nTarget: D, inhibitory regulators: C\n'
        )
        assert output_string == expected_string

    def test_model_name_setter(self, interaction_model):
        interaction_model.model_name = 'UpdatedModelName'
        assert interaction_model.model_name == 'UpdatedModelName'

    def test_size_method(self, interaction_model):
        interaction_model._interactions = [
            {'source': 'A', 'target': 'B', 'arc': 1},
            {'source': 'C', 'target': 'D', 'arc': -1}
        ]
        assert interaction_model.size() == 2

    def test_invalid_interaction_arc(self, interaction_model):
        interaction_model._interactions = [
            {'source': 'A', 'target': 'B', 'arc': 0}
        ]
        with pytest.raises(RuntimeError,match="ERROR: Invalid interaction detected. "
                                              "Source 'A', target 'B' with an unsupported value '0'."):
            interaction_model._build_multiple_interactions()

    def test_remove_self_regulated_interactions_on_init(self):
        with (patch.object(InteractionModel, '_load_sif_file'), \
                patch.object(InteractionModel, '_remove_self_regulated_interactions')
                as mock_remove_self_regulated):
            model = InteractionModel(interactions_file='dummy_file.sif', model_name='TestModel',
                                     remove_self_regulated_interactions=True)
            mock_remove_self_regulated.assert_called_once()

    def test_parsing_bidirectional_interaction(self, interaction_model):
        with patch.object(BNetworkUtil, 'parse_interaction',
                          side_effect=lambda x: {'source': x.split()[0], 'target': x.split()[-1], 'arc': 1}):
            interaction = 'A <-> B'
            if '<->' in interaction:
                line1 = interaction.replace('<->', '<-')
                line2 = interaction.replace('<->', '->')
                interaction_model._interactions.extend(
                    [BNetworkUtil.parse_interaction(line1), BNetworkUtil.parse_interaction(line2)])

            assert len(interaction_model.interactions) == 2
            assert {'source': 'A', 'target': 'B', 'arc': 1} in interaction_model.interactions

    def test_parsing_symmetric_inhibitory_interaction(self, interaction_model):
        with patch.object(BNetworkUtil, 'parse_interaction',
                          side_effect=lambda x: {'source': x.split()[0], 'target': x.split()[-1], 'arc': -1}):
            interaction = 'C |-| D'
            if '|-|' in interaction:
                line1 = interaction.replace('|-|', '|-')
                line2 = interaction.replace('|-|', '-|')
                interaction_model._interactions.extend(
                    [BNetworkUtil.parse_interaction(line1), BNetworkUtil.parse_interaction(line2)])

            assert len(interaction_model.interactions) == 2
            assert {'source': 'C', 'target': 'D', 'arc': -1} in interaction_model.interactions

    def test_parsing_partial_inhibitory_to_activating_interaction(self, interaction_model):
        with patch.object(BNetworkUtil, 'parse_interaction',
                          side_effect=lambda x: {'source': x.split()[0], 'target': x.split()[-1], 'arc': 1}):
            interaction = 'E |-> F'
            if '|->' in interaction:
                line1 = interaction.replace('|->', '->')
                line2 = interaction.replace('|->', '|-')
                interaction_model._interactions.extend(
                    [BNetworkUtil.parse_interaction(line1), BNetworkUtil.parse_interaction(line2)])

            assert len(interaction_model.interactions) == 2
            assert {'source': 'E', 'target': 'F', 'arc': 1} in interaction_model.interactions

    def test_parsing_inhibitory_to_partial_interaction(self, interaction_model):
        with patch.object(BNetworkUtil, 'parse_interaction',
                          side_effect=lambda x: {'source': x.split()[0], 'target': x.split()[-1], 'arc': -1}):
            interaction = 'G <-| H'
            if '<-|' in interaction:
                line1 = interaction.replace('<-|', '<-')
                line2 = interaction.replace('<-|', '-|')
                interaction_model._interactions.extend(
                    [BNetworkUtil.parse_interaction(line1), BNetworkUtil.parse_interaction(line2)])

            assert len(interaction_model.interactions) == 2
            assert {'source': 'G', 'target': 'H', 'arc': -1} in interaction_model.interactions

    def test_parsing_bidirectional_interaction_in_method(self):
        sif_lines = ['A <-> B']
        with patch.object(BNetworkUtil, 'read_lines_from_file', return_value=sif_lines), \
                patch.object(BNetworkUtil, 'parse_interaction',
                             side_effect=lambda x: {'source': x.split()[0], 'target': x.split()[-1],
                                                    'arc': 1}) as mock_parse_interaction:
            interaction_model = InteractionModel(interactions_file='dummy_file.sif')

            assert len(interaction_model.interactions) == 2
            mock_parse_interaction.assert_any_call('A <- B')
            mock_parse_interaction.assert_any_call('A -> B')

    def test_parsing_symmetric_inhibitory_interaction_in_method(self):
        sif_lines = ['C |-| D']
        with patch.object(BNetworkUtil, 'read_lines_from_file', return_value=sif_lines), \
                patch.object(BNetworkUtil, 'parse_interaction',
                             side_effect=lambda x: {'source': x.split()[0], 'target': x.split()[-1],
                                                    'arc': -1}) as mock_parse_interaction:
            interaction_model = InteractionModel(interactions_file='dummy_file.sif')

            assert len(interaction_model.interactions) == 2
            mock_parse_interaction.assert_any_call('C |- D')
            mock_parse_interaction.assert_any_call('C -| D')

    def test_parsing_partial_inhibitory_to_activating_interaction_in_method(self):
        sif_lines = ['E |-> F']
        with patch.object(BNetworkUtil, 'read_lines_from_file', return_value=sif_lines), \
                patch.object(BNetworkUtil, 'parse_interaction',
                             side_effect=lambda x: {'source': x.split()[0], 'target': x.split()[-1],
                                                    'arc': 1}) as mock_parse_interaction:
            interaction_model = InteractionModel(interactions_file='dummy_file.sif')

            assert len(interaction_model.interactions) == 2
            mock_parse_interaction.assert_any_call('E -> F')
            mock_parse_interaction.assert_any_call('E |- F')

    def test_parsing_inhibitory_to_partial_interaction_in_method(self):
        sif_lines = ['G <-| H']
        with patch.object(BNetworkUtil, 'read_lines_from_file', return_value=sif_lines), \
                patch.object(BNetworkUtil, 'parse_interaction',
                             side_effect=lambda x: {'source': x.split()[0], 'target': x.split()[-1],
                                                    'arc': -1}) as mock_parse_interaction:
            interaction_model = InteractionModel(interactions_file='dummy_file.sif')

            assert len(interaction_model.interactions) == 2
            mock_parse_interaction.assert_any_call('G <- H')
            mock_parse_interaction.assert_any_call('G -| H')
