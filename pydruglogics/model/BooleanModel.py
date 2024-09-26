import mpbn
from pyboolnet.file_exchange import bnet2primes
from pyboolnet.trap_spaces import compute_trap_spaces, compute_steady_states
from pydruglogics.utils.Util import Util
from pydruglogics.utils.Logger import Logger



class BooleanModel:
    def __init__(self, model=None,  file='', attractor_tool='', mutation_type='balanced', model_name='', verbosity=2,
                 equations=None, binary_equations=None):
        """
        Initializes the BooleanModel instance.
        :param model: An InteractionModel instance.
        :param file: The path to the file containing Boolean Equations in '.bnet' format.
        :param attractor_tool: The tool to be used for attractor calculation.
        :param mutation_type: The type of mutation to be performed.
        (Supported values: 'pyboolnet_stable_states', 'pyboolnet_trapspaces', 'mpbn_trapspaces')
        :param model_name: Name of the model.
        :param equations: Boolean Equations representing the model's interactions.
        :param binary: A list representing the Mutate Boolean Model in binary representation.
        """
        self._model_name = model_name
        self._boolean_equations = []
        self._updated_boolean_equations = []
        self._attractors = {}
        self._attractor_tool = attractor_tool
        self._file = file
        self._equations = equations
        self._binary_boolean_equations = [] if binary_equations is None else binary_equations
        self._is_bnet_file = False
        self._bnet_equations = ''
        self._mutation_type = mutation_type
        self._perturbations =  []
        self._global_output = 0.0
        self.verbose = verbosity
        self._logger = Logger(verbosity)

        if model is not None:
            self._init_from_model(model)
        elif self._file:
            self._init_from_bnet_file(file)
        elif self._equations is not None:
            self._init_from_equations(equations)
        else:
            raise ValueError('Please provide a model or a file for the initialization')

        self.to_binary(self._mutation_type)

    def _init_from_model(self, model) -> None:
        """
        Initialize the BooleanModel from an InteractionModel instance.
        :param model: The InteractionModel instance containing interactions.
        """
        self._logger.log('Creating Boolean Model from Interaction Model.', 3)
        self._model_name = model.model_name
        interactions = model

        for i in range(interactions.size()):
            equation = self._create_equation_from_interaction(interactions, i)
            self._boolean_equations.append(equation)

        self._updated_boolean_equations = [tuple(item) for item in self._boolean_equations]

    def _init_from_bnet_file(self, file: str) -> None:
        """
        Initialize the BooleanModel from a '.bnet' file.
        :param file: The directory of the '.bnet' file.
        """
        self._logger.log(f"Loading Boolean Model from file: {file}", 3)
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
            if line.strip().startswith('#') or line.strip().startswith('targets') or not line.strip():
                continue
            equation = line.strip()
            parsed_equation_bnet = Util.create_equation_from_bnet(equation)
            self._bnet_equations += f"{equation}\n"
            self._boolean_equations.append(parsed_equation_bnet)
            self._is_bnet_file = True

        self._updated_boolean_equations = [tuple(equation) for equation in self._boolean_equations]

    def _init_from_equations(self, equations):
        self._boolean_equations = equations
        self._updated_boolean_equations = equations

    def _calculate_attractors_mpbn(self):
        if self._is_bnet_file:
            result = self._bnet_equations
            self._is_bnet_file = False
        else:
            result = self.to_bnet_format(self._updated_boolean_equations)

        bnet_dict = Util.bnet_string_to_dict(result)
        boolean_network_mpbn = mpbn.MPBooleanNetwork(bnet_dict)
        self._attractors = list(boolean_network_mpbn.attractors())
        self._logger.log(f"\nMPBN found {len(self._attractors)} attractor(s):\n{self._attractors}", 1)

    def _calculate_attractors_pyboolnet(self, tool=''):
        if self._is_bnet_file:
            result = self._bnet_equations
            self._is_bnet_file = False
        else:
            result = self.to_bnet_format(self._updated_boolean_equations)

        primes = bnet2primes(result)
        if 'trapspaces' in tool:
            self._attractors = compute_trap_spaces(primes)
            total_nodes = len(self._updated_boolean_equations)

            updated_attractors = []
            for attractor in self._attractors:
                if len(attractor) != total_nodes:
                    full_node_set = set(target for target, _, _, _ in self._updated_boolean_equations)
                    full_attractor = {node: '*' for node in full_node_set}
                    full_attractor.update(attractor)
                    updated_attractors.append(full_attractor)
                    self._attractors = updated_attractors
                break

        elif 'stable_states' in tool:
            self._attractors = compute_steady_states(primes)
        else:
            raise ValueError('Please provide valid tool for PyBoolNet attractor calculation. '
                             'Valid values: pyboolnet_trapspaces, pyboolnet_stable_states')
        self._logger.log(f"PyBoolNet found {len(self._attractors)} attractor(s):\n{self._attractors}", 1)

    def _create_equation_from_interaction(self, interaction, interaction_index):
        """
        Create a Boolean equation from an interaction model.
        :param interaction: InteractionModel instance.
        :return: Equation dictionary with components.
        """
        activating_regulators = {}
        inhibitory_regulators = {}

        target = interaction.get_target(interaction_index)
        tmp_activating_regulators = interaction.get_activating_regulators(interaction_index)
        tmp_inhibitory_regulators = interaction.get_inhibitory_regulators(interaction_index)
        link = '' if not tmp_activating_regulators or not tmp_inhibitory_regulators else '&'

        for i, regulator in enumerate(tmp_activating_regulators):
            activating_regulators[regulator] = 1

        for i, regulator in enumerate(tmp_inhibitory_regulators):
            inhibitory_regulators[regulator] = 1

        return (target, activating_regulators, inhibitory_regulators, link,)

    def _perturb_nodes(self, node_names, effect):
        value = 0 if effect == 'inhibits' else 1

        for node in node_names:
            for i, equation in enumerate(self._updated_boolean_equations):
                target, _, _, _= equation
                if node == target:
                    new_equation = (node, {str(value): 1}, {}, '')
                    self._updated_boolean_equations[i] = new_equation
                    break

    def calculate_attractors(self, attractor_tool: str) -> None:
        """
        calculates the attractors of the boolean model. The tool for the calculation
        is based on the value of 'self.attractor_tool'.
        Values for 'self.attractor_tool' (please choose one):
        'mpbn_trapspaces', 'pyboolnet_trapspaces', 'pyboolnet_stable_states'
        :param attractor_tool:
        """
        if 'mpbn_trapspaces' in attractor_tool:
            self._calculate_attractors_mpbn()
        elif 'pyboolnet' in attractor_tool:
            self._calculate_attractors_pyboolnet(attractor_tool)
        else:
            raise ValueError('Please provide a valid attractor_tool! Valid values: mpbn_trapspaces, '
                             'pyboolnet_trapspaces, pyboolnet_stable_states')

    def calculate_global_output(self, model_outputs, normalized=True):
        """
        Use this function after you have calculated attractors with the calculate_attractors function
        in order to find the (normalized) globaloutput of the model, based on the weights of the nodes
        defined in the ModelOutputs class.
        :return: float
        """
        if not self._attractors:
            self._global_output = None
            return self._global_output

        pred_global_output = 0.0

        for attractor in self._attractors:
            for node_name, node_weight in model_outputs.model_outputs.items():
                if node_name not in attractor:
                    continue
                node_state = attractor[node_name]
                state_value = int(node_state) if node_state in [0, 1] else 0.5
                pred_global_output += state_value * node_weight

        pred_global_output /= len(self._attractors)
        if normalized:
            self._global_output = (pred_global_output - model_outputs.min_output) / (
                    model_outputs.max_output - model_outputs.min_output)
        else:
            self._global_output = pred_global_output
        return self._global_output

    def from_binary(self, binary_representation, mutation_type: str):
        """
        Updates the Boolean Equations from a binary representation based on the mutation type.
        :param binary_representation: The binary representation of the Boolean Equations as a list.
        :param mutation_type: The type of mutation can be: 'topology', 'balanced', 'mixed'
        :return: None
        """
        index = 0
        updated_equations = []
        new_link = ''

        for equation in self._updated_boolean_equations:
            target, activating, inhibitory, link = equation

            if mutation_type == 'topology':
                num_activating = len(activating)
                num_inhibitory = len(inhibitory)

                new_activating_values = binary_representation[index:index + num_activating]
                index += num_activating
                new_inhibitory_values = binary_representation[index:index + num_inhibitory]
                index += num_inhibitory

                new_activating = {key: int(val) for key, val in zip(activating.keys(), new_activating_values)}
                new_inhibitory = {key: int(val) for key, val in zip(inhibitory.keys(), new_inhibitory_values)}

                updated_equations.append((target, new_activating, new_inhibitory, link))

            elif mutation_type == 'balanced':
                if link != '':
                    link_value = binary_representation[index]
                    index += 1

                    new_link = '&' if link_value == 1 else '|'
                    updated_equations.append((target, activating, inhibitory, new_link))
                else:
                    updated_equations.append((target, activating, inhibitory, link))

            elif mutation_type == 'mixed':
                num_activating = len(activating)
                num_inhibitory = len(inhibitory)

                new_activating_values = binary_representation[index:index + num_activating]
                index += num_activating
                new_inhibitory_values = binary_representation[index:index + num_inhibitory]
                index += num_inhibitory
                if link != '':
                    link_value = binary_representation[index]
                    index += 1
                    new_link = '&' if link_value == 1 else '|'
                    new_activating = {key: val for key, val in zip(activating.keys(), new_activating_values)}
                    new_inhibitory = {key: val for key, val in zip(inhibitory.keys(), new_inhibitory_values)}
                    updated_equations.append((target, new_activating, new_inhibitory, new_link))
                else:
                    new_activating = {key: val for key, val in zip(activating.keys(), new_activating_values)}
                    new_inhibitory = {key: val for key, val in zip(inhibitory.keys(), new_inhibitory_values)}

                    updated_equations.append((target, new_activating, new_inhibitory, link))

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
            _, activating, inhibitory, link = equation

            binary_representation = []

            if mutation_type == 'topology':
                activating_values = [int(value) for value in activating.values()]
                inhibitory_values = [int(value) for value in inhibitory.values()]
                binary_representation.extend(activating_values)
                binary_representation.extend(inhibitory_values)

            elif mutation_type == 'balanced':
                if link != '':
                    binary_representation.append(1 if link == '&' else 0)
                else:
                    pass

            elif mutation_type == 'mixed':
                activating_values = [int(value) for value in activating.values()]
                inhibitory_values = [int(value) for value in inhibitory.values()]
                binary_representation.extend(activating_values)
                binary_representation.extend(inhibitory_values)
                if link == '&':
                    binary_representation.append(1)
                elif link == '|':
                    binary_representation.append(0)
                else:
                    pass

            else:
                raise ValueError('Please provide a valid mutation type. Valid types are "topology", "balanced", "mixed"')

            binary_lists.append(binary_representation)

        equation_lists = [item for sublist in binary_lists for item in sublist]
        self._binary_boolean_equations = equation_lists
        return equation_lists

    def add_perturbations(self, perturbations):
        self._perturbations = perturbations
        for drug in perturbations:
            effect = drug['effect']
            targets = drug['targets']
            self._perturb_nodes(targets, effect)

    def to_bnet_format(self, boolean_equations):
        equation_list = []

        for eq in boolean_equations:
            target, activating_regulators, inhibitory_regulators, link = eq

            target_value = f"{target}, "

            activation_terms = [regulator for regulator, value in activating_regulators.items() if value == 1]
            inhibition_terms = [f"!{regulator}" for regulator, value in inhibitory_regulators.items() if value == 1]

            activation_expression = " | ".join(activation_terms)
            inhibition_expression = " | ".join(inhibition_terms)

            if activation_expression and inhibition_expression:
                combined_expression = f"{activation_expression} {link} {inhibition_expression}"
            elif activation_expression or inhibition_expression:
                combined_expression = activation_expression or inhibition_expression
            else:
                combined_expression = '0'

            equation_line = f"{target_value.strip()} {combined_expression.strip()}"
            modified_line = equation_line.replace('(', '').replace(')', '')
            equation_list.append(modified_line)

        final_equation_list = '\n'.join(equation_list)
        return final_equation_list

    def print(self):
        equation_list = ''
        for eq in self._updated_boolean_equations:
            equation = ''
            target, activating, inhibitory, link = eq

            activating_nodes = [node for node, value in activating.items() if value == 1]
            inhibitory_nodes = [node for node, value in inhibitory.items() if value == 1]

            if activating_nodes and inhibitory_nodes:
                activating_part = ' or '.join(activating_nodes)
                inhibitory_part = ' or '.join(inhibitory_nodes)
                equation += f"{target} *= ({activating_part}) {link} not ({inhibitory_part})"
            elif activating_nodes and not inhibitory_nodes:
                activating_part = ' or '.join(activating_nodes)
                equation += f"{target} *= {activating_part}"
            elif inhibitory_nodes and not activating_nodes:
                inhibitory_part = ' or '.join(inhibitory_nodes)
                equation += f"{target} *= not {inhibitory_part}"
            else:
                equation += f"{target} *= 0"

            equation_list += equation
            equation_list += '\n'

        print(equation_list)

    def apply_perturbations(self, perturbations):
        """
        Apply a list of perturbations to the current Boolean model.
        :param perturbations: A list of perturbations where each perturbation is a dictionary with 'targets' and 'effect'.
        """
        for drug in perturbations:
            effect = drug['effect']
            targets = drug['targets']
            self._perturb_nodes(targets, effect)

    def clone(self):
        return BooleanModel(
            model_name=self._model_name,
            attractor_tool=self._attractor_tool,
            mutation_type=self._mutation_type,
            verbosity=self.verbose,
            equations=self._updated_boolean_equations.copy(),
            binary_equations=self._binary_boolean_equations.copy()
        )

    def reset_attractors(self) -> None:
        self._attractors = []

    def has_attractors(self) -> bool:
        return bool(self._attractors)

    def has_stable_states(self) -> bool:
        return bool(self.get_stable_states())

    def has_global_output(self) -> bool:
        return bool(self.global_output)

    def get_stable_states(self) -> object:
        return [state for state in self._attractors if '*' not in state]

    @property
    def mutation_type(self) -> str:
        return self._mutation_type

    @property
    def perturbations(self):
        return self._perturbations

    @property
    def global_output(self):
        return self._global_output

    @property
    def updated_boolean_equations(self):
        return self._updated_boolean_equations

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

    @property
    def boolean_equations(self):
        return self._boolean_equations

    @model_name.setter
    def model_name(self, model_name: str) -> None:
        self._model_name = model_name

    @updated_boolean_equations.setter
    def updated_boolean_equations(self, updated_boolean_equations: dict) -> None:
        self._updated_boolean_equations = updated_boolean_equations

    @binary_boolean_equations.setter
    def binary_boolean_equations(self, binary_boolean_equations) -> None:
        self._binary_boolean_equations = binary_boolean_equations
