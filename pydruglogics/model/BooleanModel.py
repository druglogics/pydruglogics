import copy
import os
import logging
import numpy as np
import mpbn
from pyboolnet.file_exchange import bnet2primes
from pyboolnet.trap_spaces import compute_trap_spaces, compute_steady_states
from pydruglogics.utils.BNetworkUtil import BNetworkUtil


class BooleanModel:
    def __init__(self, model=None,  file=None, attractor_tool='mpbn', attractor_type='stable_states',
                 mutation_type='topology', model_name='', cloned_equations=None, cloned_binary_equations=None):
        """
        Initializes the BooleanModel instance.
        :param model: An InteractionModel instance.
        :param file: The path to the file containing Boolean Equations in '.bnet' format.
        :param attractor_tool: The tool to be used for attractor calculation. Possible values: 'mpbn', 'pyboolnet'.
        :param attractor_type: The type of the attractor calculation. Possible values: 'stable_states', 'trapspaces'.
        :param mutation_type: The type of mutation to be performed. Possible values: 'topology', 'binary', 'balanced'.
        :param model_name: Name for the model.
        :param cloned_equations: Boolean Equations representing the model's interactions. (only for cloning)
        :param cloned_binary_equations: A list representing the Mutate Boolean Model in binary representation.
        (only for cloning)
        """
        self._model_name = model_name
        self._boolean_equations = []
        self._binary_boolean_equations = np.array([])
        self._attractors = {}
        self._attractor_tool = attractor_tool
        self._attractor_type = attractor_type
        self._mutation_type = mutation_type
        self._global_output = 0.0
        self._is_bnet_file = False
        self._bnet_equations = ''

        if attractor_tool not in ['mpbn', 'pyboolnet'] or attractor_type not in ['stable_states', 'trapspaces']:
            raise ValueError("Invalid attractor tool or type. Use 'mpbn' or 'pyboolnet' for attractor_tool, and "
                             "'stable_states' or 'trapspaces' for attractor_type.")

        if mutation_type not in ['topology', 'mixed', 'balanced']:
            raise ValueError("Invalid mutation type. Use 'topology' or 'mixed' or 'balanced' for mutation_type.")

        try:
            if model is not None: # init from InteractionModel instance
                self._init_from_model(model)
                self.to_binary(self._mutation_type)
            elif file is not None:
                self._init_from_bnet_file(file) # init from .bnet file
                self.to_binary(self._mutation_type)
            elif cloned_binary_equations is not None:
                self._init_from_equations(cloned_equations, cloned_binary_equations) # for cloning
            else:
                raise ValueError('Initialization failed: Please provide a model or a file for the Boolean Model.')
        except Exception as e:
            logging.error(f"Error occurred during the initialization: {str(e)}")
            raise

    def _init_from_model(self, model) -> None:
        """
        Initialize the BooleanModel from an InteractionModel instance.
        :param model: The InteractionModel instance containing interactions.
        """
        self._model_name = model.model_name

        self._boolean_equations.extend(self._create_equation_from_interaction(model, i) for i in range(model.size()))

        logging.info('Boolean Model from Interaction Model is created.')

    def _init_from_bnet_file(self, file: str) -> None:
        """
        Initialize the BooleanModel from a '.bnet' file.
        :param file: The directory of the '.bnet' file.
        :return: None
        """
        logging.debug(f"Loading Boolean Model from file: {file}")
        try:
            with open(file, 'r') as model_file:
                lines = model_file.readlines()

            if BNetworkUtil.get_file_extension(file) != 'bnet':
                raise ValueError('The file extension has to be .bnet format.')

            self._boolean_equations = []
            self._model_name = os.path.splitext(os.path.basename(file))[0]

            for line in lines:
                if line.strip().startswith('#') or line.strip().startswith('targets') or not line.strip():
                    continue
                equation = line.strip()
                parsed_equation_bnet = BNetworkUtil.create_equation_from_bnet(equation)
                self._bnet_equations += f"{equation}\n"
                self._boolean_equations.append(parsed_equation_bnet)
                self._is_bnet_file = True

            logging.info('Boolean Model from .bnet file is created.')

        except IOError as e:
            logging.error(f"Error reading file: {str(e)}")
            raise

    def _init_from_equations(self, equations, binary_equations):
        """
        Init for cloning.
        :param equations: Boolean Equations for cloning.
        :param binary_equations: Binary Equations for cloning.
        :return: None
        """
        self._boolean_equations = equations
        self._binary_boolean_equations = np.array(binary_equations)

    def _get_equations_in_bnet_format(self):
        if self._is_bnet_file:
            self._is_bnet_file = False
            return self._bnet_equations
        return BNetworkUtil.to_bnet_format(self._boolean_equations)

    def _calculate_attractors_mpbn(self, attractor_type):
        """
        Calculate attractors using MPBN based on attractor_type.
        :param attractor_type: Attractor type for calculation  ('stable_states' or 'trapspaces').
        :return: None
        """
        equations_bnet_dict = BNetworkUtil.bnet_string_to_dict(self._get_equations_in_bnet_format())
        boolean_network_mpbn = mpbn.MPBooleanNetwork(equations_bnet_dict)

        self._attractors = list(boolean_network_mpbn.fixedpoints() if attractor_type == 'stable_states'
                                else boolean_network_mpbn.attractors())

        logging.debug(f"MPBN found {len(self._attractors)} attractor(s):\n{self._attractors}")

    def _calculate_attractors_pyboolnet(self, attractor_type):
        """
        Calculate attractors using PyBoolNet based on attractor_type.
        :param attractor_type: Attractor type for calculation ('stable_states' or 'trapspaces').
        :return: None
        """
        primes = bnet2primes(self._get_equations_in_bnet_format())

        if attractor_type == 'trapspaces':
            self._attractors = compute_trap_spaces(primes)
            all_node_in_model = {target for target, _, _, _ in self._boolean_equations}

            self._attractors = [
                {**{node: '*' for node in all_node_in_model}, **attractor}
                if len(attractor) < len(all_node_in_model) else attractor
                for attractor in self._attractors
            ]

        elif attractor_type == 'stable_states':
            self._attractors = compute_steady_states(primes)

        logging.debug(f"PyBoolNet found {len(self._attractors)} attractor(s):\n{self._attractors}")

    def _create_equation_from_interaction(self, interaction, interaction_index):
        """
        Create a Boolean equation from an InteractionModel.
        :param interaction: An interaction of the InteractionModel.
        :param interaction_index: Index of an interaction.
        :return: Equation dictionary with target, regulators, operators.
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
        """
        Apply a perturbation to the given nodes by modifying their Boolean equations.
        :param node_names: A list of node names to be perturbed.
        :param effect: The type of perturbation effect ('inhibits' sets the value to 0, others set it to 1).
        :return: None
        """
        value = 0 if effect == 'inhibits' else 1

        for node in node_names:
            for i, equation in enumerate(self._boolean_equations):
                target, _, _, _= equation
                if node == target:
                    new_equation = (node, {str(value): 1}, {}, '')
                    self._boolean_equations[i] = new_equation
                    break

    def calculate_attractors(self, attractor_tool, attractor_type) -> None:
        """
        Calculate attractors based on the chosen attractor_tool and attractor_type.
        :param attractor_tool: Tool for attractor calculation. Possible values: 'mpbn', 'pyboolnet'.
        :param attractor_type: Type of attractor. Possible values: 'stable_states', 'trapspaces'.
        :return: None
        """
        tool_methods = {
            'mpbn': self._calculate_attractors_mpbn,
            'pyboolnet': self._calculate_attractors_pyboolnet
        }

        if attractor_tool in tool_methods:
            tool_methods[attractor_tool](attractor_type)
        else:
            raise ValueError("Please provide a valid attractor tool and type. Valid tools: 'mpbn', 'pyboolnet'. "
                             "Valid types: 'stable_states', 'trapspaces'.")


    def calculate_global_output(self, model_outputs, normalized=True):
        """
        Calculates the (normalized) global output of the model.
        :param model_outputs: An instance containing node weights defined in the ModelOutputs class.
        :param normalized: Whether to normalize the global output.
        :return: float
        """
        if not self._attractors:
            self._global_output = None
            logging.debug('No attractors were found')
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
        :param mutation_type: The type of mutation can be: 'topology', 'balanced', 'mixed'.
        :return: Updated Boolean Equations as a list.
        """
        index = 0
        new_link = ''

        for i, (target, activating, inhibitory, link) in enumerate(self._boolean_equations):
            num_activating = len(activating)
            num_inhibitory = len(inhibitory)

            if mutation_type in ['topology', 'mixed'] and len(activating) + len(inhibitory) <= 1:
                continue

            elif mutation_type in ['topology', 'mixed']:
                new_activating_values = binary_representation[index:index + num_activating]
                index += num_activating
                new_inhibitory_values = binary_representation[index:index + num_inhibitory]
                index += num_inhibitory

                if num_activating > 0 and num_inhibitory > 0:
                    if (all(val == 0 for val in new_activating_values) and
                            all(val == 0 for val in new_inhibitory_values)):
                        new_activating_values[0] = 1

                elif num_activating > 0 and num_inhibitory == 0:
                    if all(val == 0 for val in new_activating_values):
                        new_activating_values[0] = 1

                elif num_inhibitory > 0 and num_activating == 0:
                    if all(val == 0 for val in new_inhibitory_values):
                        new_inhibitory_values[0] = 1

                new_activating = dict(zip(activating.keys(), new_activating_values))
                new_inhibitory = dict(zip(inhibitory.keys(), new_inhibitory_values))

                if mutation_type == 'mixed' and link != '':
                    link_value = binary_representation[index]
                    index += 1
                    link = '&' if link_value == 1 else '|'

                self._boolean_equations[i] = (target, new_activating, new_inhibitory, link)

            elif mutation_type == 'balanced':
                if link != '':
                    link_value = binary_representation[index]
                    index += 1
                    link = '&' if link_value == 1 else '|'
                    self._boolean_equations[i] = (target, activating, inhibitory, link)

        return self._boolean_equations

    def to_binary(self, mutation_type: str):
        """
        Converts the Boolean Equations to a binary representation. The representation is based on the mutation type.
        :param mutation_type: The type of mutation can be: 'topology', 'balanced', 'mixed'
        :return: The binary representation of the Boolean Equations as a list.
        """
        binary_representation = []

        for _, activating, inhibitory, link in self._boolean_equations:
            if mutation_type in ['topology', 'mixed'] and len(activating) + len(inhibitory) > 1:
                binary_representation.extend(int(val) for val in activating.values())
                binary_representation.extend(int(val) for val in inhibitory.values())
                if mutation_type == 'mixed' and link != '':
                    binary_representation.append(1 if link == '&' else 0)

            elif mutation_type == 'balanced' and link != '':
                binary_representation.append(1 if link == '&' else 0)

        self._binary_boolean_equations = np.array(binary_representation)
        return self._binary_boolean_equations

    def add_perturbations(self, perturbations):
        """
        Adds perturbations to the Boolean Model.
        :param perturbations: A list of Perturbations.
        :return: None
        """
        for perturbation in perturbations:
            self._perturb_nodes(perturbation['targets'], perturbation['effect'])

    def print(self):
        equations = []
        link_operator_map = {'&': 'and', '|': 'or', '': ''}

        for eq in self._boolean_equations:
            target, activating, inhibitory, link = eq
            activating_nodes = [node for node, value in activating.items() if value == 1]
            inhibitory_nodes = [node for node, value in inhibitory.items() if value == 1]

            if activating_nodes and inhibitory_nodes:
                activating_part = ' or '.join(activating_nodes)
                inhibitory_part = ' or '.join(inhibitory_nodes)
                converted_link = link_operator_map.get(link, link)
                equation = f"{target} *= ({activating_part}) {converted_link} not ({inhibitory_part})"
            elif activating_nodes:
                activating_part = ' or '.join(activating_nodes)
                equation = f"{target} *= ({activating_part})"
            elif inhibitory_nodes:
                inhibitory_part = ' or '.join(inhibitory_nodes)
                equation = f"{target} *= not ({inhibitory_part})"
            else:
                equation = f"{target} *= 0"

            equations.append(equation)

        print('\n'.join(equations))

    def clone(self):
        return BooleanModel(
            model_name=self._model_name,
            attractor_tool=self._attractor_tool,
            attractor_type=self._attractor_type,
            mutation_type=self._mutation_type,
            cloned_equations=self._boolean_equations.copy(),
            cloned_binary_equations=self._binary_boolean_equations.copy()
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
        return [state for state in self._attractors if '*' not in state.values()]

    @property
    def mutation_type(self) -> str:
        return self._mutation_type

    @property
    def global_output(self):
        return self._global_output

    @property
    def boolean_equations(self):
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

    @property
    def attractor_type(self) -> str:
        return self._attractor_type

    @model_name.setter
    def model_name(self, model_name: str) -> None:
        self._model_name = model_name

    @boolean_equations.setter
    def boolean_equations(self, boolean_equations: dict) -> None:
        self._boolean_equations = boolean_equations

    @binary_boolean_equations.setter
    def binary_boolean_equations(self, binary_boolean_equations) -> None:
        self._binary_boolean_equations = binary_boolean_equations
