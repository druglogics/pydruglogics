import pytest
from unittest.mock import Mock, patch, mock_open
from pydruglogics import BooleanModel
from pydruglogics.utils.BNetworkUtil import BNetworkUtil


class TestBooleanModel:

    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.model_name = 'test_model'
        model.size.return_value = 2
        model.get_target.side_effect = ['A', 'B']
        model.get_activating_regulators.side_effect = [['ACT1'], ['ACT2']]
        model.get_inhibitory_regulators.side_effect = [[], ['INHIB1']]
        return model

    @pytest.fixture
    def boolean_model(self, mock_model):
        return BooleanModel(model=mock_model)

    # Init
    def test_init_from_file_invalid_extension(self):
        with pytest.raises(IOError):
            with patch('builtins.open', mock_open(read_data="data")):
                BooleanModel(file='model.txt')

    def test_init_from_file_read_error(self):
        with patch('builtins.open', side_effect=IOError("File not found")):
            with pytest.raises(IOError):
                BooleanModel(file='model.bnet')

    def test_init_from_file(self):
        with patch('builtins.open', mock_open(read_data="A, B, 0\nB, C, 1")), \
                patch.object(BNetworkUtil, 'get_file_extension', return_value='bnet'):
            model = BooleanModel(file='model.bnet')
            assert model.model_name == 'model'
            assert len(model.boolean_equations) == 2

    def test_init_from_model(self, mock_model):
        boolean_model = BooleanModel(model=mock_model)
        assert boolean_model.model_name == 'test_model'
        assert len(boolean_model.boolean_equations) == 2

    def test_init_from_invalid_params(self):
        with pytest.raises(ValueError):
            BooleanModel()

    def test_init_from_file_invalid_extension(self):
        with pytest.raises(ValueError, match="The file extension has to be .bnet format."):
            with patch('builtins.open', mock_open(read_data="data")), \
                    patch.object(BNetworkUtil, 'get_file_extension', return_value='txt'):
                BooleanModel(file='model.txt')

    def test_init_from_bnet_file_skips_lines(self):
        file_content = "# Comment line\n" \
                       "targets, A\n" \
                       "\n" \
                       "A, B & C"

        with patch('builtins.open', mock_open(read_data=file_content)), \
                patch.object(BNetworkUtil, 'get_file_extension', return_value='bnet'):
            model = BooleanModel(file='model.bnet')
        assert model.boolean_equations == [('A', {'B': 1, 'C': 1}, {}, '')]

    def test_invalid_mutation_type(self):
        with pytest.raises(ValueError,
                           match="Invalid mutation type. Use 'topology' or 'mixed' or 'balanced' for mutation_type."):
            BooleanModel(mutation_type='invalid_type')

    def test_binary_conversion_mixed(self, boolean_model):
        binary_representation = [0, 1, 1]
        activating = {'A': 1}
        inhibitory = {'B': 1}
        link = '&'
        num_activating = len(activating)
        num_inhibitory = len(inhibitory)
        new_activating_values = binary_representation[:num_activating]
        new_inhibitory_values = binary_representation[num_activating:num_activating + num_inhibitory]
        index = num_activating + num_inhibitory
        new_link = '&' if binary_representation[index] == 1 else '|'
        new_activating = dict(zip(activating.keys(), new_activating_values))
        new_inhibitory = dict(zip(inhibitory.keys(), new_inhibitory_values))

        assert new_link == '&'
        assert new_activating == {'A': 0}
        assert new_inhibitory == {'B': 1}


    # Global output
    def test_calculate_global_output_no_attractors(self, boolean_model):
        mock_outputs = Mock(min_output=0, max_output=10)
        boolean_model._attractors = []
        result = boolean_model.calculate_global_output(mock_outputs)
        assert result is None

    def test_calculate_global_output_normalized(self, boolean_model):
        boolean_model._attractors = [{'A': 1, 'B': 0}, {'A': 0, 'B': 1}]
        mock_outputs = Mock()
        mock_outputs.model_outputs = {'A': 1, 'B': 2}
        mock_outputs.min_output = 0
        mock_outputs.max_output = 10
        result = boolean_model.calculate_global_output(mock_outputs)
        assert result == 0.15

    def test_calculate_global_output_skip_missing_nodes(self, boolean_model):
        boolean_model._attractors = [{'A': 1, 'B': 0}, {'A': 0, 'B': 1}]
        mock_outputs = Mock()
        mock_outputs.model_outputs = {'A': 1, 'C': -1}
        mock_outputs.min_output = -1
        mock_outputs.max_output = 1
        result = boolean_model.calculate_global_output(mock_outputs)
        assert result == 0.75

    def test_calculate_global_output_skip_nodes_not_in_attractor(self, boolean_model):
        boolean_model._attractors = [{'A': 1, 'B': 0}]
        mock_outputs = Mock()
        mock_outputs.model_outputs = {'A': 1, 'B': -1, 'C': 0.6}
        mock_outputs.min_output = -1
        mock_outputs.max_output = 1
        result = boolean_model.calculate_global_output(mock_outputs)
        assert result == 1

    def test_calculate_global_output_without_normalization(self, boolean_model):
        boolean_model._attractors = [{'A': 1, 'B': 0}, {'A': 0, 'B': 1}]
        mock_outputs = Mock()
        mock_outputs.model_outputs = {'A': 1, 'B': 2}
        mock_outputs.min_output = 0
        mock_outputs.max_output = 10

        result = boolean_model.calculate_global_output(mock_outputs, normalized=False)
        assert result == 1.5


    # Attractor calculation
    def test_invalid_attractor_tool_and_type(self):
        with pytest.raises(ValueError, match="Invalid attractor tool or type. Use 'mpbn' or 'pyboolnet' for "
                                             "attractor_tool, and 'stable_states' or 'trapspaces' for attractor_type."):
            BooleanModel(attractor_tool='invalid_tool', attractor_type='invalid_type')

    def test_calculate_attractors_resets_bnet_flag_mpbn(self, boolean_model):
        boolean_model._bnet_equations = "A, C | D\nB, A & !D"
        boolean_model._is_bnet_file = True

        with patch('mpbn.MPBooleanNetwork', autospec=True) as MockMPBooleanNetwork:
            mock_mpbn_instance = MockMPBooleanNetwork.return_value
            mock_mpbn_instance.fixedpoints.return_value = [{'A': 1, 'B': 0}]
            boolean_model._calculate_attractors_mpbn('stable_states')
            assert boolean_model._is_bnet_file is False
            assert boolean_model.attractors == [{'A': 1, 'B': 0}]

    def test_calculate_attractors_resets_bnet_flag_pyboolnet(self, boolean_model):
        boolean_model._bnet_equations = "A, B | C"
        boolean_model._is_bnet_file = True

        with patch('pyboolnet.file_exchange.bnet2primes', return_value="primes"), \
                patch('pyboolnet.trap_spaces.compute_steady_states', return_value=[{'A': 1}]):
            boolean_model._calculate_attractors_pyboolnet('stable_states')
            assert boolean_model._is_bnet_file is False
            assert boolean_model.attractors == [{'A': 0, 'B': 0, 'C': 0}, {'A': 1, 'B': 0, 'C': 1},
                                                {'A': 1, 'B': 1, 'C': 0},{'A': 1, 'B': 1, 'C': 1}]

    def test_calculate_attractors_mpbn_stable_states(self, boolean_model):
        with patch('mpbn.MPBooleanNetwork', autospec=True) as MockMPBooleanNetwork:
            mock_mpbn_instance = MockMPBooleanNetwork.return_value
            mock_mpbn_instance.fixedpoints.return_value = [{'A': 1, 'B': 0}]
            boolean_model._calculate_attractors_mpbn('stable_states')
            assert boolean_model.attractors == [{'A': 1, 'B': 0}]

    def test_calculate_attractors_pyboolnet_stable_states(self, boolean_model):
        with patch('pyboolnet.file_exchange.bnet2primes', return_value="primes"), \
                patch('pyboolnet.trap_spaces.compute_steady_states', return_value=[{'A': 1, 'B': 1}]):
            boolean_model._calculate_attractors_pyboolnet('stable_states')
            assert boolean_model.attractors == [{'A': 1, 'ACT1': 1, 'ACT2': 0, 'B': 0, 'INHIB1': 1},
                                                 {'A': 0, 'ACT1': 0, 'ACT2': 0, 'B': 0, 'INHIB1': 1},
                                                 {'A': 1, 'ACT1': 1, 'ACT2': 0, 'B': 0, 'INHIB1': 0},
                                                 {'A': 0, 'ACT1': 0, 'ACT2': 0, 'B': 0, 'INHIB1': 0},
                                                 {'A': 0, 'ACT1': 0, 'ACT2': 1, 'B': 0, 'INHIB1': 1},
                                                 {'A': 1, 'ACT1': 1, 'ACT2': 1, 'B': 0, 'INHIB1': 1},
                                                 {'A': 0, 'ACT1': 0, 'ACT2': 1, 'B': 1, 'INHIB1': 0},
                                                 {'A': 1, 'ACT1': 1, 'ACT2': 1, 'B': 1, 'INHIB1': 0}]

    def test_calculate_attractors_invalid_tool(self, boolean_model):
        with pytest.raises(ValueError,
                           match="Please provide a valid attractor tool and type. Valid tools: 'mpbn', 'pyboolnet'. "
                             "Valid types: 'stable_states', 'trapspaces'."):
            boolean_model.calculate_attractors('invalid_tool', 'stable_states')


    def test_calculate_attractors_mpbn(self, boolean_model):
        boolean_model._bnet_equations = 'A, C | D\nB, A & !D'
        bnet_dict = {
            "A": "C | D",
            "B": "A & !D"
            }

        with patch('mpbn.MPBooleanNetwork', autospec=True) as MockMPBooleanNetwork, \
                patch.object(BNetworkUtil, 'bnet_string_to_dict', return_value=bnet_dict):

            mock_mpbn_instance = MockMPBooleanNetwork.return_value
            mock_mpbn_instance.attractors.return_value = [{'A': 1, 'B': 0}]
            boolean_model._calculate_attractors_mpbn('trapspaces')
            MockMPBooleanNetwork.assert_called_once_with(bnet_dict)

            assert len(boolean_model.attractors) == 1
            assert boolean_model.attractors == [{'A': 1, 'B': 0}]

    def test_calculate_attractors_pyboolnet_trapspaces(self, boolean_model):
        with patch('pyboolnet.trap_spaces.compute_trap_spaces', return_value=[{'A': 0, 'B': '*'}]), \
             patch('pyboolnet.file_exchange.bnet2primes', return_value="primes"):
            boolean_model._calculate_attractors_pyboolnet('trapspaces')
            assert len(boolean_model.attractors) == 8

    def test_calculate_attractors_call_mpbn(self, boolean_model):
        with patch('mpbn.MPBooleanNetwork', autospec=True) as MockMPBooleanNetwork:
            mock_mpbn_instance = MockMPBooleanNetwork.return_value
            mock_mpbn_instance.attractors.return_value = [{'A': 1, 'B': 0}]
            boolean_model.calculate_attractors('mpbn', 'trapspaces')
            MockMPBooleanNetwork.assert_called_once()

    def test_calculate_attractors_calls_pyboolnet(self, boolean_model):
        boolean_model._bnet_equations = "A, B & C"

        with patch('pyboolnet.file_exchange.bnet2primes', return_value="primes"), \
                patch('pyboolnet.trap_spaces.compute_trap_spaces', return_value=[{'A': 1, 'B': 0}]):
            boolean_model.calculate_attractors('pyboolnet', 'trapspaces')
            assert boolean_model.attractors == [{'A': 1, 'ACT1': 1, 'ACT2': 0, 'B': 0, 'INHIB1': 0},
                                                 {'A': 0, 'ACT1': 0, 'ACT2': 0, 'B': 0, 'INHIB1': 0},
                                                 {'A': 0, 'ACT1': 0, 'ACT2': 0, 'B': 0, 'INHIB1': 1},
                                                 {'A': 1, 'ACT1': 1, 'ACT2': 0, 'B': 0, 'INHIB1': 1},
                                                 {'A': 1, 'ACT1': 1, 'ACT2': 1, 'B': 0, 'INHIB1': 1},
                                                 {'A': 0, 'ACT1': 0, 'ACT2': 1, 'B': 0, 'INHIB1': 1},
                                                 {'A': 0, 'ACT1': 0, 'ACT2': 1, 'B': 1, 'INHIB1': 0},
                                                 {'A': 1, 'ACT1': 1, 'ACT2': 1, 'B': 1, 'INHIB1': 0}]

    def test_bnet_equations_reset_in_calculate_attractors_mpbn(self, boolean_model):
        boolean_model._bnet_equations = "A, C | D\nB, A & !D"
        boolean_model._is_bnet_file = True

        with patch('mpbn.MPBooleanNetwork', autospec=True) as MockMPBooleanNetwork:
            mock_mpbn_instance = MockMPBooleanNetwork.return_value
            mock_mpbn_instance.attractors.return_value = [{'A': 1, 'B': 0}]
            boolean_model._calculate_attractors_mpbn('stabel_states')
            assert boolean_model._is_bnet_file is False

    def test_bnet_equations_reset_in_calculate_attractors_pyboolnet(self, boolean_model):
        boolean_model._bnet_equations = "A, C | D\nB, A & !D"
        boolean_model._is_bnet_file = True

        with patch('pyboolnet.file_exchange.bnet2primes', return_value="primes"), \
                patch('pyboolnet.trap_spaces.compute_trap_spaces', return_value=[{'A': 0, 'B': 1}]):
            boolean_model._calculate_attractors_pyboolnet('pyboolnet_trapspaces')
            assert boolean_model._is_bnet_file is False

    # From_binary
    def test_from_binary_update_first_inhibitory_zero_values(self, boolean_model):
        boolean_model._boolean_equations = [('Target', {}, {'Inhibitor1': 0, 'Inhibitor2': 0}, '&')]
        binary_representation = [0, 0]
        boolean_model.from_binary(binary_representation, 'topology')

        updated_equation = boolean_model.boolean_equations[0]
        target, new_activating, new_inhibitory, link = updated_equation

        assert new_inhibitory == {'Inhibitor1': 1, 'Inhibitor2': 0}

    def test_from_binary_update_first_activating_zero_values(self, boolean_model):
        boolean_model._boolean_equations = [('Target', {'Reg1': 0, 'Reg2': 0}, {}, '&')]
        binary_representation = [0, 0]
        boolean_model.from_binary(binary_representation, 'topology')
        updated_equation = boolean_model.boolean_equations[0]
        _, new_activating, _, _ = updated_equation
        assert new_activating == {'Reg1': 1, 'Reg2': 0}

    def test_from_binary_balanced(self, boolean_model):
        boolean_model._boolean_equations = [('A', {'B': 1}, {}, '&'), ('B', {}, {'A': 1}, '|')]
        binary = [0, 1]
        boolean_model.from_binary(binary, 'balanced')
        assert boolean_model.boolean_equations[0][3] == '|'
        assert boolean_model.boolean_equations[1][3] == '&'

    def test_from_binary_not_new_link_balanced(self, boolean_model):
        boolean_model._boolean_equations = [('A', {'B': 1}, {}, '')]
        binary = [0, 1]
        boolean_model.from_binary(binary, 'balanced')
        assert boolean_model.boolean_equations[0][3] == ''

    def test_from_binary_mixed_multiple_activating_and_inhibitory(self, boolean_model):
        binary_representation = [1, 0, 1, 1, 0]
        boolean_model._boolean_equations = [('B', {'X': 1, 'Z': 1}, {'Y': 1, 'W': 0}, '&')]
        boolean_model.from_binary(binary_representation, 'mixed')
        updated_equation = boolean_model.boolean_equations[0]
        target, new_activating, new_inhibitory, new_link = updated_equation

        assert target == 'B'
        assert new_activating == {'X': 1, 'Z': 0}
        assert new_inhibitory == {'Y': 1, 'W': 1}
        assert new_link == '|'

    def test_from_binary_mixed_only_activating_nodes(self, boolean_model):
        binary_representation = [0, 1]
        boolean_model._boolean_equations = [('C', {'X': 1, 'Z': 0}, {}, '')]

        boolean_model.from_binary(binary_representation, 'mixed')
        updated_equation = boolean_model.boolean_equations[0]
        target, new_activating, new_inhibitory, new_link = updated_equation

        assert target == 'C'
        assert new_activating == {'X': 0, 'Z': 1}
        assert new_inhibitory == {}
        assert new_link == ''

    def test_from_binary_mixed_edge_case_all_zeroes(self, boolean_model):
        binary_representation = [0, 0, 0, 0, 0]
        boolean_model._boolean_equations = [('D', {'X': 1, 'Z': 1}, {'Y': 1, 'W': 1}, '&')]
        boolean_model.from_binary(binary_representation, 'mixed')
        updated_equation = boolean_model.boolean_equations[0]
        target, new_activating, new_inhibitory, new_link = updated_equation

        assert target == 'D'
        assert new_activating == {'X': 1, 'Z': 0}
        assert new_inhibitory == {'Y': 0, 'W': 0}
        assert new_link == '|'

    def test_from_binary_mixed_link_update(self, boolean_model):
        binary_representation = [1, 0, 1, 0]
        boolean_model._boolean_equations = [('E', {'X': 1}, {'Y': 1, 'Z': 1}, '&')]

        boolean_model.from_binary(binary_representation, 'mixed')
        updated_equation = boolean_model.boolean_equations[0]
        target, new_activating, new_inhibitory, new_link = updated_equation

        assert target == 'E'
        assert new_activating == {'X': 1}
        assert new_inhibitory == {'Y': 0, 'Z':1}
        assert new_link == '|'

    def test_from_binary_mixed_link_not_update(self, boolean_model):
        binary_representation = [1, 0, 0]
        boolean_model._boolean_equations = [('E', {'X': 1}, {'Y': 0}, '&')]
        boolean_model.from_binary(binary_representation, 'mixed')

        updated_equation = boolean_model.boolean_equations[0]
        target, new_activating, new_inhibitory, new_link = updated_equation

        assert target == 'E'
        assert new_activating == {'X': 1}
        assert new_inhibitory == {'Y': 0}
        assert new_link == '|'

    def test_from_binary_mixed_one_not_update(self, boolean_model):
        binary_representation = [0, 1]
        boolean_model._boolean_equations = [('E', {'X': 1}, {}, '&')]
        boolean_model.from_binary(binary_representation, 'mixed')

        updated_equation = boolean_model.boolean_equations[0]
        target, new_activating, new_inhibitory, new_link = updated_equation

        assert target == 'E'
        assert new_activating == {'X': 1}
        assert new_inhibitory == {}
        assert new_link == '&'

    def test_update_equations_not_topology(self, boolean_model):
        binary_representation = [1, 0, 1]
        boolean_model._boolean_equations = [('Target', {'X': 0, 'Y': 1}, {'Z': 1}, '&')]
        boolean_model.from_binary(binary_representation, 'topology')
        updated_equation = boolean_model.boolean_equations[0]
        target, new_activating, new_inhibitory, link = updated_equation

        assert target == 'Target'
        assert new_activating == {'X': 1, 'Y': 0}
        assert new_inhibitory == {'Z': 1}


    def test_update_equations_not_all_zero_topology(self, boolean_model):
        binary_representation = [0, 0, 0, 0]
        boolean_model._boolean_equations = [('Target', {'A': 0, 'B': 1}, {'C': 0, 'D': 1}, '&')]
        boolean_model.from_binary(binary_representation, 'topology')
        updated_equation = boolean_model.boolean_equations[0]
        target, new_activating, new_inhibitory, link = updated_equation

        assert target == 'Target'
        assert new_activating == {'A': 1, 'B': 0}
        assert new_inhibitory == {'C': 0, 'D': 0}


    # To_binary
    def test_to_binary_mixed_link_appended(self, boolean_model):
        boolean_model._boolean_equations = [('Target', {'Activator1': 1}, {'Inhibitor1': 1}, '&')]
        binary_representation = boolean_model.to_binary('mixed')

        assert binary_representation[-1] == 1

    def test_to_binary_balanced_with_link_and(self, boolean_model):
        boolean_model._boolean_equations = [('A', {}, {}, '&')]
        binary_representation = boolean_model.to_binary('balanced')
        assert binary_representation == [1]

    def test_to_binary_balanced_with_link_or(self, boolean_model):
        boolean_model._boolean_equations = [('A', {}, {}, '|')]
        binary_representation = boolean_model.to_binary('balanced')
        assert binary_representation == [0]

    # Perturbation
    def test_add_perturbations(self, boolean_model):
        boolean_model._boolean_equations = [('A', {'B': 1}, {}, ''), ('B', {}, {'A': 1}, '')]
        perturbations = [{'effect': 'inhibits', 'targets': ['A']}]
        boolean_model.add_perturbations(perturbations)
        assert boolean_model._boolean_equations[0][1] == {'0': 1}


    # Getter, setter, print
    def test_attractor_tool_property(self, boolean_model):
        boolean_model._attractor_tool = 'mpbn'
        assert boolean_model.attractor_tool == 'mpbn'

    def test_attractor_type_property(self, boolean_model):
        boolean_model._attractor_type = 'trapspaces'
        assert boolean_model.attractor_type == 'trapspaces'

    def test_has_global_output_with_value(self, boolean_model):
        boolean_model._global_output = 0.7
        assert boolean_model.has_global_output() is True

    def test_has_global_output_without_value(self, boolean_model):
        boolean_model._global_output = 0.0
        assert boolean_model.has_global_output() is False

    def test_boolean_equations_setter(self, boolean_model):
        new_equations = [('X', {'Y': 1}, {}, '&'), ('Y', {'Z': 1}, {}, '|')]
        boolean_model.boolean_equations = new_equations
        assert boolean_model.boolean_equations == new_equations

    def test_reset_attractors(self, boolean_model):
        boolean_model._attractors = ['state1', 'state2']
        boolean_model.reset_attractors()
        assert not boolean_model._attractors

    def test_has_attractors(self, boolean_model):
        boolean_model._attractors = ['state1']
        assert boolean_model.has_attractors() is True
        boolean_model._attractors = []
        assert boolean_model.has_attractors() is False

    def test_get_stable_states(self, boolean_model):
        boolean_model._attractors = [{'A': 1, 'B': '*'}, {'A': 1, 'B': 1}]
        assert boolean_model.get_stable_states() == [{'A': 1, 'B': 1}]

    def test_boolean_equations_property(self, boolean_model):
        boolean_model._boolean_equations = ['equation1', 'equation2']
        assert boolean_model.boolean_equations == ['equation1', 'equation2']

    def test_attractors_property(self, boolean_model):
        boolean_model._attractors = ['state1', 'state2']
        assert boolean_model.attractors == ['state1', 'state2']

    def test_clone(self, boolean_model):
        boolean_model._model_name = "test_clone"
        clone = boolean_model.clone()
        assert clone.model_name == "test_clone"
        assert clone is not boolean_model

    def test_model_name_setter(self, boolean_model):
        boolean_model.model_name = "new_name"
        assert boolean_model.model_name == "new_name"

    def test_binary_boolean_equations_setter(self, boolean_model):
        binary_eq = [0, 1, 0]
        boolean_model.binary_boolean_equations = binary_eq
        assert boolean_model.binary_boolean_equations == binary_eq

    def test_global_output_property(self, boolean_model):
        boolean_model._global_output = 0.5
        assert boolean_model.global_output == 0.5

    def test_mutation_type_property(self, boolean_model):
        boolean_model._mutation_type = "topology"
        assert boolean_model.mutation_type == "topology"

    def test_has_stable_states(self, boolean_model):
        boolean_model._attractors = [{'A': 1, 'B': 1}, {'A': 0, 'B': '*'}]
        assert boolean_model.has_stable_states() is True

        boolean_model._attractors = [{'A': '*', 'B': '*'}]
        assert boolean_model.has_stable_states() is False

    def test_print_method(self, boolean_model):
        boolean_model._boolean_equations = [
            ('A', {'B': 1, 'C': 1}, {'D': 1}, '&'),
            ('B', {'E': 1}, {}, ''),
            ('C', {}, {'F': 1}, '')]

        expected_output = (
            "A *= (B or C) and not (D)\n"
            "B *= (E)\n"
            "C *= not (F)")

        with pytest.MonkeyPatch.context() as m:
            printed_output = []
            m.setattr('builtins.print', lambda x: printed_output.append(x))
            boolean_model.print()

            assert ''.join(printed_output) == expected_output

    def test_print_method_no_activators_or_inhibitors(self, boolean_model):
        boolean_model._boolean_equations = [('A', {}, {}, '')]
        expected_output = "A *= 0"

        with patch('builtins.print') as mock_print:
            boolean_model.print()

        mock_print.assert_called_with(expected_output)
