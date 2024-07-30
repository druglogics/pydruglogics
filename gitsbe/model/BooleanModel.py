import tempfile
import random
import biolqm
import mpbn
import pyboolnet
from pyboolnet.file_exchange import bnet2primes
from pyboolnet.trap_spaces import compute_trap_spaces
from gitsbe.utils.Util import Util


class BooleanModel:
    def __init__(self, model=None, file='', attractor_tool='',  model_name=''):
        """
        Initializes the BooleanModel instance.
        :param model: An InteractionModel instance.
        :param file: The path to the file containing Boolean Equations in '.bnet' format.
        :param attractor_tool: Tool to be used for attractor calculation.
        (Supported values 'biolqm_trapspaces', 'biolqm_stable_states', 'mpbn_trapspaces', 'pyboolnet_trapspaces')
        :param model_name: Name of the model.
        """
        self._model_name = model_name
        self._boolean_equations = []
        self._updated_boolean_equations = []
        self._attractors = {}
        self._attractor_tool = attractor_tool
        self._fitness = 0
        self._file = file
        self._binary_boolean_equations = []
        self._is_bnet_file = False
        self._bnet_equations = ''

        if model is not None:
            self.init_from_model(model)
        elif self._file:
            self.init_from_bnet_file(file)
        else:
            raise ValueError('Please provide a model or a file for the initialization')

    def init_from_model(self, model) -> None:
        """
        Initialize the BooleanModel from an InteractionModel instance.
        :param model: The InteractionModel instance containing interactions.
        """
        self._model_name = model.model_name
        interactions = model

        for i in range(interactions.size()):
            equation = self.create_equation_from_interaction(interactions, i)
            self._boolean_equations.append(equation)

        self._updated_boolean_equations = [tuple(item) for item in self._boolean_equations]

    def init_from_bnet_file(self, file: str) -> None:
        """
        Initialize the BooleanModel from a '.bnet' file.
        :param file: The directory of the '.bnet' file.
        """
        print(f"Loading Boolean model from file: {file}")
        try:
            with open(file, 'r') as model_file:
                lines = model_file.readlines()

        except IOError as e:
            raise IOError(f"Error reading file: {e}")

        if Util.get_file_extension(file) != 'bnet':
            raise IOError('ERROR: The extension needs to be .bnet!')

        self._boolean_equations = []
        self._model_name = file.rsplit('.', 1)[0]

        for line in lines:
            if line.strip().startswith('#') or not line.strip():
                continue
            equation = line.strip()
            parsed_equation_bnet = self._create_equation_from_bnet(equation)
            self._bnet_equations += f"{equation}\n"
            self._boolean_equations.append(parsed_equation_bnet)
            self._is_bnet_file = True

        self._updated_boolean_equations = [tuple(item) for item in self._boolean_equations]

    def calculate_attractors(self, attractor_tool: str) -> None:
        """
        calculates the attractors of the boolean model. The tool for the calculation
        is based on the value of 'self.attractor_tool'.
        Values for 'self.attractor_tool' (please choose one):
        'biolqm_trapspaces', 'biolqm_stable_states', 'mpbn_trapspaces', 'pyboolnet_trapspaces'
        :param attractor_tool:
        """
        if 'biolqm' in attractor_tool:
            self._calculate_attractors_biolqm()
        if 'mpbn' in attractor_tool:
            self._calculate_attractors_mpbn()
        else:
            self._calculate_attractors_pyboolnet()

    def _calculate_attractors_biolqm(self) -> str:
        if self._is_bnet_file:
            result = self._bnet_equations
        else:
            result = self.to_bnet_format(self._updated_boolean_equations)

        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.bnet') as temp:
            temp.write(result)
            temp_name = temp.name
        lqm = biolqm.load(temp_name)

        if 'stable' in self._attractor_tool:
            self._attractors = biolqm.fixpoints(lqm)
        elif 'trapspace' in self._attractor_tool:
            self._attractors = biolqm.trapspace(lqm)

        if self._attractors:
            return f"BioLQM found {len(self._attractors)} attractors."
        return 'BioLQM found no attractors.'

    def _calculate_attractors_mpbn(self) -> str:
        if self._is_bnet_file:
            result = self._bnet_equations
            self._is_bnet_file = False
        else:
            result = self.to_bnet_format(self._updated_boolean_equations)

        bnet_dict = Util.bnet_string_to_dict(result)
        boolean_network_mpbn = mpbn.MPBooleanNetwork(bnet_dict)
        self._attractors = list(boolean_network_mpbn.attractors())
        return f"MPBN found {len(self._attractors)} attractors."

    def _calculate_attractors_pyboolnet(self) -> str:
        if self._is_bnet_file:
            result = self._bnet_equations
            self._is_bnet_file = False
        else:
            result = self.to_bnet_format(self._updated_boolean_equations)

        primes = bnet2primes(result)
        self._attractors = compute_trap_spaces(primes)
        return f"PyBoolNet found {len(self._attractors)} attractors."

    def get_index_of_equation(self, node_name: str) -> int:
        """
        Gets the index of the equation for a given node name.
        :param node_name: The name of the node.
        :return: The index of the equation or -1 if not found.
        """
        for index, equation in enumerate(self._updated_boolean_equations):
            target, *_ = equation
            if target == node_name:
                return index
        return -1

    def from_binary(self, binary_representation, mutation_type: str):
        """
        Updates the Boolean Equations from a binary representation based on the mutation type.
        :param binary_representation: The binary representation of the Boolean Equations as a list.
        :param mutation_type: The type of mutation can be: 'topology', 'balanced', 'mixed'
        :return: None
        """
        index = 0
        updated_equations = []

        for equation in self._updated_boolean_equations:
            target, activating, inhibitory, act_operators, inhib_operators, link = equation

            if mutation_type == 'topology':
                num_activating = len(activating)
                num_inhibitory = len(inhibitory)

                new_activating_values = binary_representation[index:index + num_activating]
                index += num_activating
                new_inhibitory_values = binary_representation[index:index + num_inhibitory]
                index += num_inhibitory

                new_activating = {key: int(val) for key, val in zip(activating.keys(), new_activating_values)}
                new_inhibitory = {key: int(val) for key, val in zip(inhibitory.keys(), new_inhibitory_values)}

                updated_equations.append((target, new_activating, new_inhibitory, act_operators, inhib_operators, link))

            elif mutation_type == 'balanced':
                link_value = binary_representation[index]
                index += 1

                new_link = 'and' if link_value == 1 else 'or'
                updated_equations.append((target, activating, inhibitory, act_operators, inhib_operators, new_link))

            elif mutation_type == 'mixed':
                num_activating = len(activating)
                num_inhibitory = len(inhibitory)

                new_activating_values = binary_representation[index:index + num_activating]
                index += num_activating
                new_inhibitory_values = binary_representation[index:index + num_inhibitory]
                index += num_inhibitory
                link_value = binary_representation[index]
                index += 1

                new_activating = {key: val for key, val in zip(activating.keys(), new_activating_values)}
                new_inhibitory = {key: val for key, val in zip(inhibitory.keys(), new_inhibitory_values)}
                new_link = 'and' if link_value == 1 else 'or'

                updated_equations.append((target, new_activating, new_inhibitory,
                                          act_operators, inhib_operators, new_link))

        self._updated_boolean_equations = updated_equations
        return self._updated_boolean_equations

    def to_binary(self, mutation_type: str):
        """
        Converts the Boolean Equations to a binary representation. It is based on the mutation type.
        :param mutation_type: The type of mutation can be: 'topology', 'balanced', 'mixed'
        :return: The binary representation of the Boolean Equations as a list.
        """
        binary_lists = []

        for equation in self._updated_boolean_equations:
            _, activating, inhibitory, _, _, link = equation

            binary_representation = []

            if mutation_type == 'topology':
                activating_values = [int(value) for value in activating.values()]
                inhibitory_values = [int(value) for value in inhibitory.values()]
                binary_representation.extend(activating_values)
                binary_representation.extend(inhibitory_values)

            elif mutation_type == 'balanced':
                binary_representation.append(1 if link == 'and' else 0)

            elif mutation_type == 'mixed':
                activating_values = [int(value) for value in activating.values()]
                inhibitory_values = [int(value) for value in inhibitory.values()]
                binary_representation.extend(activating_values)
                binary_representation.extend(inhibitory_values)
                binary_representation.append(1 if link == 'and' else 0)

            binary_lists.append(binary_representation)

        equation_lists = [item for sublist in binary_lists for item in sublist]
        self._binary_boolean_equations = equation_lists
        return equation_lists


    def generate_mutated_lists(self, num_mutations, num_mutations_per_list):
        list_length = len(self._binary_boolean_equations)
        mutated_lists = []

        for _ in range(num_mutations):
            mutated_list = self._binary_boolean_equations.copy()
            for _ in range(num_mutations_per_list):
                index_to_mutate = random.randint(0, list_length - 1)
                mutated_list[index_to_mutate] = 1 - mutated_list[index_to_mutate]
            mutated_lists.append(mutated_list)

        return mutated_lists

    def create_equation_from_interaction(self, interaction, interaction_index):
        """
        Create a Boolean equation from an interaction model.
        :param interaction: InteractionModel instance.
        :return: Equation dictionary with components.
        """
        activating_regulators = {}
        inhibitory_regulators = {}
        operators_activating_regulators = []
        operators_inhibitory_regulators = []

        target = interaction.get_target(interaction_index)
        tmp_activating_regulators = interaction.get_activating_regulators(interaction_index)
        tmp_inhibitory_regulators = interaction.get_inhibitory_regulators(interaction_index)
        link = '' if not tmp_activating_regulators or not tmp_inhibitory_regulators else 'and'

        for i, regulator in enumerate(tmp_activating_regulators):
            activating_regulators[regulator] = 1
            if i < (len(tmp_activating_regulators) - 1):
                operators_activating_regulators.append('or')

        for i, regulator in enumerate(tmp_inhibitory_regulators):
            inhibitory_regulators[regulator] = 1
            if i < (len(tmp_inhibitory_regulators) - 1):
                operators_inhibitory_regulators.append('or')

        interaction_tuple = (
            target,
            activating_regulators,
            inhibitory_regulators,
            operators_activating_regulators,
            operators_inhibitory_regulators,
            link,
        )

        return interaction_tuple

    def create_equation_from_bnet(self, equation_str):
        activating_regulators = {}
        inhibitory_regulators = {}
        operators_activating_regulators = []
        operators_inhibitory_regulators = []
        link = ''

        arg = (equation_str.strip()
               .replace(', ', ' *= ')
               .replace('!', ' not ')
               .replace('&', ' and ')
               .replace('|', ' or '))
        split_arg = arg.split()
        target = split_arg.pop(0)
        if split_arg.pop(0) != '*=':
            raise ValueError("Equation must start with ','")

        before_not = True

        for regulator in split_arg:
            if regulator == 'not':
                before_not = not before_not
            elif regulator in ('and', 'or'):
                if before_not:
                    operators_activating_regulators.append(regulator)
                    link = 'and'
                else:
                    operators_inhibitory_regulators.append(regulator)
                    link = 'or'
            elif regulator in ('(', ')'):
                continue
            else:
                if before_not:
                    activating_regulators[regulator] = 1
                else:
                    inhibitory_regulators[regulator] = 1
                    before_not = True

        link = '' if not activating_regulators or not inhibitory_regulators else 'and'

        interaction_tuple = (
            target,
            activating_regulators,
            inhibitory_regulators,
            operators_activating_regulators,
            operators_inhibitory_regulators,
            link
        )

        return interaction_tuple

    def to_bnet_format(self, boolean_equations):
        equation_list = []

        for eq in boolean_equations:
            target, activating_regulators, inhibitory_regulators, _, _, link = eq

            target_value = f"{target}, "
            link_operator = {'and': '&', 'or': '|'}.get(link, '')

            activation_terms = [regulator for regulator, value in activating_regulators.items() if value == 1]
            inhibition_terms = [f"!{regulator}" for regulator, value in inhibitory_regulators.items() if value == 1]

            activation_expression = " | ".join(activation_terms)
            inhibition_expression = " | ".join(inhibition_terms)

            if activation_expression and inhibition_expression:
                combined_expression = f"{activation_expression} {link_operator} {inhibition_expression}"
            elif activation_expression or inhibition_expression:
                combined_expression = activation_expression or inhibition_expression
            else:
                combined_expression = '0'

            equation_line = f"{target_value.strip()} {combined_expression.strip()}"
            modified_line = equation_line.replace('(', '').replace(')', '')
            equation_list.append(modified_line)

        final_equation_list = '\n'.join(equation_list)
        print('Equations:')
        print(final_equation_list)
        return final_equation_list

    def _create_equation_from_interaction(self, interaction, interaction_index):
        activating_regulators = {}
        inhibitory_regulators = {}
        operators_activating_regulators = []
        operators_inhibitory_regulators = []

        target = interaction.get_target(interaction_index)
        tmp_activating_regulators = interaction.get_activating_regulators(interaction_index)
        tmp_inhibitory_regulators = interaction.get_inhibitory_regulators(interaction_index)

        link = '' if not tmp_activating_regulators or not tmp_inhibitory_regulators else 'and'

        for i, regulator in enumerate(tmp_activating_regulators):
            activating_regulators[regulator] = 1
            if i < (len(tmp_activating_regulators) - 1):
                operators_activating_regulators.append('or')

        for i, regulator in enumerate(tmp_inhibitory_regulators):
            inhibitory_regulators[regulator] = 1
            if i < (len(tmp_inhibitory_regulators) - 1):
                operators_inhibitory_regulators.append('or')

        return (target, activating_regulators, inhibitory_regulators,
                operators_activating_regulators, operators_inhibitory_regulators, link)

    def _create_equation_from_bnet(self, equation_str):
        activating_regulators = {}
        inhibitory_regulators = {}
        operators_activating_regulators = []
        operators_inhibitory_regulators = []
        link = ''

        arg = (equation_str.strip().replace(', ', ' *= ').replace('!', ' not ')
               .replace('&', ' and ').replace('|', ' or '))
        split_arg = arg.split()
        target = split_arg.pop(0)
        if split_arg.pop(0) != '*=':
            raise ValueError("Equation must start with '*='")

        before_not = True

        for regulator in split_arg:
            if regulator == 'not':
                before_not = not before_not
            elif regulator in ('and', 'or'):
                if before_not:
                    operators_activating_regulators.append(regulator)
                else:
                    operators_inhibitory_regulators.append(regulator)
            elif regulator in ('(', ')'):
                continue
            else:
                if before_not:
                    activating_regulators[regulator] = 1
                else:
                    inhibitory_regulators[regulator] = 1
                    before_not = True

        link = '' if not activating_regulators or not inhibitory_regulators else 'and'

        return (target, activating_regulators, inhibitory_regulators,
                operators_activating_regulators, operators_inhibitory_regulators, link)

    def reset_attractors(self) -> None:
        self._attractors = []

    def has_attractors(self) -> bool:
        return bool(self._attractors)

    def has_stable_states(self) -> bool:
        return bool(self.get_stable_states())

    def get_stable_states(self) -> object:
        return [state for state in self._attractors if '-' not in state]

    @property
    def boolean_equations(self) -> object:
        return self._boolean_equations

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def attractors(self) -> object:
        return self._attractors

    @property
    def binary_boolean_equations(self):
        return self._binary_boolean_equations

    @property
    def attractor_tool(self) -> str:
        return self._attractor_tool

    @model_name.setter
    def model_name(self, model_name: str) -> None:
        self._model_name = model_name
