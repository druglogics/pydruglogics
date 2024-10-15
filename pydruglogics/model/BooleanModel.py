import os
import logging
from colomoto import minibn
import mpbn
import biolqm
from pyboolnet.file_exchange import bnet2primes
from pyboolnet.trap_spaces import compute_trap_spaces, compute_steady_states
from pydruglogics.utils.Util import Util



class BooleanModel:
    def __init__(self, model=None,  file='', attractor_tool='', mutation_type='balanced', model_name='',
                 equations=None, binary_equations=None):
        """
        Initializes the BooleanModel instance.
        :param model: An InteractionModel instance.
        :param file: The path to the file containing Boolean Equations in '.bnet' format.
        :param attractor_tool: The tool to be used for attractor calculation.
        :param mutation_type: The type of mutation to be performed.
        (Supported values: 'pyboolnet_stable_states', 'pyboolnet_trapspaces', 'mpbn_fixpoints',
        'mpbn_trapspaces', 'biolqm_fixpoints', 'biolqm_trapspaces')
        :param model_name: Name of the model.
        :param equations: Boolean Equations representing the model's interactions.
        :param binary_equations: A list representing the Mutate Boolean Model in binary representation.
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

        try:
            if model is not None: # init from InteractionModel instance
                self._init_from_model(model)
                self.to_binary(self._mutation_type)
            elif self._file:
                self._init_from_bnet_file(file) # init from .bnet file
                self.to_binary(self._mutation_type)
            elif self._equations is not None:
                self._init_from_equations(equations) # for cloning
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
        interactions = model

        for i in range(interactions.size()):
            equation = self._create_equation_from_interaction(interactions, i)
            self._boolean_equations.append(equation)

        self._updated_boolean_equations = [tuple(item) for item in self._boolean_equations]
        logging.info('Boolean Model from Interaction Model is created.')

    def _init_from_bnet_file(self, file: str) -> None:
        """
        Initialize the BooleanModel from a '.bnet' file.
        :param file: The directory of the '.bnet' file.
        """
        logging.debug(f"Loading Boolean Model from file: {file}")
        try:
            with open(file, 'r') as model_file:
                lines = model_file.readlines()

            if Util.get_file_extension(file) != 'bnet':
                raise ValueError('The file extension has to be .bnet format.')

            self._boolean_equations = []
            self._model_name = os.path.splitext(os.path.basename(file))[0]

            for line in lines:
                if line.strip().startswith('#') or line.strip().startswith('targets') or not line.strip():
                    continue
                equation = line.strip()
                parsed_equation_bnet = Util.create_equation_from_bnet(equation)
                self._bnet_equations += f"{equation}\n"
                self._boolean_equations.append(parsed_equation_bnet)
                self._is_bnet_file = True

            self._updated_boolean_equations = [tuple(equation) for equation in self._boolean_equations]
            logging.info('Boolean Model from .bnet file is created.')

        except IOError as e:
            logging.error(f"Error reading file: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error occurred during the initialization from file: {str(e)}")
            raise

    def _init_from_equations(self, equations):
        self._boolean_equations = equations
        self._updated_boolean_equations = equations

    def _calculate_attractors_biolqm(self, tool=''):
        if self._is_bnet_file:
            result = self._bnet_equations
            self._is_bnet_file = False
        else:
            result = Util.to_bnet_format(self._updated_boolean_equations)

        bnet_dict = Util.bnet_string_to_dict(result)
        boolean_network_minibn = minibn.BooleanNetwork(bnet_dict)
        network_biolqm = boolean_network_minibn.to_biolqm()
        if 'fixpoints' in tool:
            calculated_attractors = biolqm.fixpoints(network_biolqm)
        elif 'trapspaces' in tool:
            calculated_attractors = biolqm.trapspaces(network_biolqm)
        else:
            logging.error('Attractor tool is not provided.')
            raise ValueError('Please provide valid tool for BioLQM attractor calculation. '
                             'Valid values: biolqm_trapspaces, biolqm_fixpoints')

        if calculated_attractors:
            self._attractors = [
                {key: ('*' if value == 254 else value) for key, value in attractor.items()}
                for attractor in calculated_attractors
            ]
        else:
            self._attractors = []

        logging.debug(f"BioLQM found {len(self._attractors)} attractor(s):\n{self._attractors}")

    def _calculate_attractors_mpbn(self, tool=''):
        if self._is_bnet_file:
            result = self._bnet_equations
            self._is_bnet_file = False
        else:
            result = Util.to_bnet_format(self._updated_boolean_equations)

        bnet_dict = Util.bnet_string_to_dict(result)
        boolean_network_mpbn = mpbn.MPBooleanNetwork(bnet_dict)
        if 'fixpoints' in tool:
            self._attractors = list(boolean_network_mpbn.fixedpoints())
        elif 'trapspaces' in tool:
            self._attractors = list(boolean_network_mpbn.attractors())
        logging.debug(f"\nMPBN found {len(self._attractors)} attractor(s):\n{self._attractors}")

    def _calculate_attractors_pyboolnet(self, tool=''):
        if self._is_bnet_file:
            result = self._bnet_equations
            self._is_bnet_file = False
        else:
            result = Util.to_bnet_format(self._updated_boolean_equations)

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
        logging.debug(f"PyBoolNet found {len(self._attractors)} attractor(s):\n{self._attractors}")

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
        'mpbn_trapspaces',  'mpbn_fixepoints', 'pyboolnet_trapspaces',
        'pyboolnet_stable_states','biolqm_trapspaces',  'biolqm_fixpoints'
        :param attractor_tool:
        """
        if 'mpbn' in attractor_tool:
            self._calculate_attractors_mpbn(attractor_tool)
        elif 'pyboolnet' in attractor_tool:
            self._calculate_attractors_pyboolnet(attractor_tool)
        elif 'biolqm' in attractor_tool:
            self._calculate_attractors_biolqm(attractor_tool)
        else:
            raise ValueError("Please provide a valid attractor tool. Valid tools: 'mpbn_trapspaces', "
                             "'pyboolnet_trapspaces', 'pyboolnet_stable_states', 'biolqm_fixpoints', 'biolqm_trapspaces'")


    def calculate_global_output(self, model_outputs, normalized=True):
        """
        Use this function after you have calculated attractors to find the (normalized) global output of the model.
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
        updated_equations = []
        new_link = ''

        for target, activating, inhibitory, link in self._updated_boolean_equations:
            num_activating = len(activating)
            num_inhibitory = len(inhibitory)

            if mutation_type in ['topology', 'mixed'] and len(activating) + len(inhibitory) <= 1:
                updated_equations.append((target, activating, inhibitory, link))
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

                if mutation_type ==  'mixed' and link != '':
                    link_value = binary_representation[index]
                    index += 1
                    link = '&' if link_value == 1 else '|'
                    new_activating = dict(zip(activating.keys(), new_activating_values))
                    new_inhibitory = dict(zip(inhibitory.keys(), new_inhibitory_values))

                updated_equations.append((target, new_activating, new_inhibitory, link))

            elif mutation_type == 'balanced':
                if link != '':
                    link_value = binary_representation[index]
                    index += 1

                    new_link = '&' if link_value == 1 else '|'
                    updated_equations.append((target, activating, inhibitory, new_link))
                else:
                    updated_equations.append((target, activating, inhibitory, link))

            else:
                raise ValueError("Please provide a valid mutation type: 'balanced', 'topology',  'mixed'")

        self._updated_boolean_equations = updated_equations
        return self._updated_boolean_equations

    def to_binary(self, mutation_type: str):
        """
        Converts the Boolean Equations to a binary representation. It is based on the mutation type.
        :param mutation_type: The type of mutation can be: 'topology', 'balanced', 'mixed'
        :return: The binary representation of the Boolean Equations as a list.
        """
        binary_lists = []

        for _, activating, inhibitory, link in self._updated_boolean_equations:
            num_activating = len(activating)
            num_inhibitory = len(inhibitory)
            binary_representation = []

            if mutation_type in ['topology', 'mixed'] and len(activating) + len(inhibitory) <= 1:
                continue

            elif mutation_type in ['topology', 'mixed'] and (num_activating + num_inhibitory > 1):
                binary_representation = [int(val) for val in activating.values()]
                binary_representation.extend(int(val) for val in inhibitory.values())

                if mutation_type == 'mixed' and link != '':
                    binary_representation.append(1 if link=='&' else 0)

                binary_lists.append(binary_representation)

            elif mutation_type == 'balanced':
                if link != '':
                    binary_representation.append(1 if link == '&' else 0)
                else:
                    pass
                binary_lists.append(binary_representation)

            else:
                raise ValueError("Please provide a valid mutation type: 'topology', 'balanced', 'mixed'")

        self._binary_boolean_equations = [bit for binary_rep in binary_lists for bit in binary_rep]
        return self._binary_boolean_equations

    def add_perturbations(self, perturbations):
        """
        Adds perturbations to the Boolean Model.
        :param perturbations: A list of Perturbations.
        """
        self._perturbations = perturbations
        for drug in perturbations:
            effect = drug['effect']
            targets = drug['targets']
            self._perturb_nodes(targets, effect)

    def print(self):
        equations = []
        link_operator_map = {'&': 'and', '|': 'or', '': ''}

        for eq in self._updated_boolean_equations:
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
            mutation_type=self._mutation_type,
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
        return [state for state in self._attractors if '*' not in state.values()]

    @property
    def mutation_type(self) -> str:
        return self._mutation_type

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
