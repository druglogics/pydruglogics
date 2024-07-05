import random
import biolqm
import mpbn
import tempfile
import pyboolnet
from pyboolnet.file_exchange import bnet2primes
from pyboolnet.trap_spaces import compute_trap_spaces
from gitsbe.input.ModelOutputs import ModelOutputs
from gitsbe.model.BooleanEquation import BooleanEquation
from gitsbe.utils.Util import Util


class BooleanModel:
    def __init__(self, model=None, file='', attractor_tool='', model_name=''):
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
        self._attractors = []
        self._attractor_tool = attractor_tool
        self._fitness = 0
        self._file = file
        self._binary_boolean_equations = []

        if model is not None:
            self.init_from_model(model)
        elif self._file != '' and not model:
            self.init_from_file(file, attractor_tool)
        else:
            print('Please provide only a model or a file')

    def init_from_model(self, model) -> None:
        """
        Initialize the BooleanModel from an InteractionModel instance.
        :param model: The InteractionModel instance containing interactions.
        """
        self._model_name = 'model.model_name'
        interactions = model

        for i in range(interactions.size()):
            equation = BooleanEquation(interactions, i)
            self._boolean_equations.append(equation)

    def init_from_file(self, file: str, attractor_tool: str) -> None:
        """
        Initialize the BooleanModel from a '.bnet' file.
        :param file: The directory of the '.bnet' file.
        :param attractor_tool:  Tool to be used for attractor calculation.
        """
        print(f"Loading Boolean model from file: {file}")
        try:
            with open(file, 'r') as model_file:
                lines = model_file.readlines()

        except IOError as e:
            print(f"Error reading file: {e}")
            return

        if Util.get_file_extension(file) != 'bnet':
            raise IOError('ERROR: The extension needs to be .bnet!')

        self._boolean_equations = []
        self._model_name = file.rsplit('.', 1)[0]

        for _, line in enumerate(lines[0:], 1):
            if line.strip().startswith('#') or not line.strip():
                continue
            equation_bool_net = line.strip()
            equation_boolean_net = (equation_bool_net.replace(",", " *=")
                                    .replace("&", " and ")
                                    .replace("|", " or ")
                                    .replace("!", " not ")
                                    .replace(" 1 ", " true ")
                                    .replace(" 0 ", " false "))
            self._boolean_equations.append(BooleanEquation(equation_boolean_net))

    def calculate_attractors(self, attractor_tool: str) -> None:
        """
        calculates the attractors of the boolean model. The tool for the calculation
        is based on the value of 'self.attractor_tool'.
        Values for 'self.attractor_tool' (please choose one):
        'biolqm_trapspaces', 'biolqm_stable_states', 'mpbn_trapspaces', 'pyboolnet_trapspaces'

        :param attractor_tool:
        """
        if 'biolqm' in attractor_tool:
            self.calculate_attractors_biolqm(attractor_tool)
        if 'mpbn' in attractor_tool:
            self.calculate_attractors_mpbn()
        else:
            self.calculate_attractors_pyboolnet()

    def calculate_attractors_biolqm(self, attractor_tool) -> str:
        result = ''.join(equation.to_bnet_format() for equation in self._boolean_equations)
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.bnet') as temp:
            temp.write(result)
            temp_name = temp.name
        lqm = biolqm.load(temp_name)

        if 'stable' in attractor_tool:
            self._attractors = biolqm.fixpoints(lqm)
            if self._attractors:
                return f"BioLQM found {len(self._attractors)} stable states."
        elif 'trapspace' in attractor_tool:
            self._attractors = biolqm.trapspace(lqm)
            if self._attractors:
                return f"BioLQM found {len(self._attractors)} trap spaces."

        return 'BioLQM found no attractors.'

    def calculate_attractors_mpbn(self) -> str:
        result = ''.join(equation.to_bnet_format() for equation in self._boolean_equations)
        bnet_dictionary = Util.bnet_string_to_dict(result)
        boolean_network_mpbn = mpbn.MPBooleanNetwork(bnet_dictionary)
        self._attractors = list(boolean_network_mpbn.attractors())
        return f"MPBN found {len(self._attractors)} trap spaces."

    def calculate_attractors_pyboolnet(self) -> str:
        result = ''.join(equation.to_bnet_format() for equation in self._boolean_equations)
        primes = bnet2primes(result)
        self._attractors = pyboolnet.trap_spaces.compute_trap_spaces(primes)
        return f"PyBoolNet found {len(self._attractors)} trap spaces."

    def update_boolean_model_balance(self, solution) -> None:
        """
        Updates the BooleanModel based on the updated balance binary representation
        for both activating and inhibiting values.
        :param solution: A list of values used to update the Boolean Equations.
        """
        smaller_lists = [solution[i:i + 7] for i in range(0, len(solution), 7)]
        eq_to_be_converted = self._boolean_equations
        for iter, sol in enumerate(smaller_lists):
            eq_to_be_converted[iter].modify_link_from_list(sol)

    def update_boolean_model_both(self, solution) -> None:
        """
        Updates the BooleanModel based on the updated mixed binary representation
        for both activating and inhibiting values.
        :param solution: A list of values used to update the Boolean Equations.
        """
        smaller_lists = [solution[i:i + 7] for i in range(0, len(solution), 7)]
        eq_to_be_converted = self._boolean_equations
        for iter, sol in enumerate(smaller_lists):
            eq_to_be_converted[iter].modify_activating_values_from_list(sol)
            eq_to_be_converted[iter].modify_inhibitory_values_from_list(sol)
            eq_to_be_converted[iter].modify_link_from_list(sol)

    def update_boolean_model_topology(self, solution) -> None:
        """
        Updates the BooleanModel based on the updated topology binary representation
        for both activating and inhibiting values.
        :param solution: A list of values used to update the Boolean Equations.
        """
        smaller_lists = [solution[i:i + 6] for i in range(0, len(solution), 6)]
        eq_to_be_converted = self._boolean_equations
        for iter, sol in enumerate(smaller_lists):
            eq_to_be_converted[iter].modify_activating_values_from_list(sol)
            eq_to_be_converted[iter].modify_inhibitory_values_from_list(sol)

    def calculate_global_output(self) -> float:
        """
        Use this function after you have calculated attractors with the calculate_attractors function
        in order to find the normalized globaloutput of the model, based on the weights of the nodes
        defined in the ModelOutputs class.
        :return: float
        """
        outputs = ModelOutputs.get_instance()
        global_output = 0

        for attractor in self._attractors:
            for node in outputs.model_outputs:
                node_name = node['node_name']
                node_weight = node['weight']
                if node_name:
                    node_state = attractor[node_name]
                    state_value = int(node_state) if node_state in [0, 1] else 0.5
                    global_output += state_value * node_weight

        global_output /= len(self._attractors)
        return (global_output - outputs.min_output) / (outputs.max_output - outputs.min_output)

    def get_index_of_equation(self, node_name: str) -> int:
        """
        Gets the index of the equation for a given node name.
        :param node_name: The name of the node.
        :return: The index of the equation or -1 if not found.
        """
        for index, equation in enumerate(self._boolean_equations):
            if equation.target == node_name:
                return index
        return -1

    def to_binary(self, mutation_type: str):
        """
        Converts the Boolean Equations to a binary representation. It is based on the mutation type.
        :param mutation_type: The type of mutation can be: 'topology', 'balanced', 'mixed'
        :return: The binary representation of the Boolean Equations as a list.
        """
        binary_lists = []

        for equation in self._boolean_equations:
            activating = equation.get_values_activating_regulators()
            inhibitory = equation.get_values_inhibitory_regulators()
            link = equation.link

            if len(activating) != 3:
                activating += [-1] * (3 - len(activating))
            if len(inhibitory) != 3:
                inhibitory += [-1] * (3 - len(inhibitory))

            binary_representation = []

            if mutation_type == 'topology':
                binary_representation = activating + inhibitory

            elif mutation_type == 'balanced':
                if link == 'and':
                    binary_representation = [0, 0, 0, 0, 0, 0, 1]
                elif link == 'or':
                    binary_representation = [0, 0, 0, 0, 0, 0, 0]
                else:
                    binary_representation = [0, 0, 0, 0, 0, 0, -1]
            elif mutation_type == 'mixed':
                binary_representation = activating + inhibitory
                if link == 'and':
                    binary_representation.append(1)
                elif link == 'or':
                    binary_representation.append(0)
                else:
                    binary_representation.append(-1)

            binary_lists.append(binary_representation)

        merged_list = [item for sublist in binary_lists for item in sublist]
        self._binary_boolean_equations = binary_lists
        for binary_list in binary_lists:
            print('is this here')
            print(binary_list)
            print('end is this here')
        return merged_list

    def topology_mutations(self, number_of_mutations: int):
        """
        Introduces mutations to topology, removing regulators of nodes (not all regulators for any node)
        :param number_of_mutations:
        :return: The list of updated Boolean equations.
        """
        for _ in range(number_of_mutations):
            random_equation_index = random.randint(0, len(self._boolean_equations) - 1)
            orig = self._boolean_equations[random_equation_index].get_boolean_equation()
            self._boolean_equations[random_equation_index].mutate_regulator()
            if self._boolean_equations[random_equation_index].get_boolean_equation() != orig:
                print(f"Exchanging equation {random_equation_index}\n\t{orig}\n\t"
                      f"{self._boolean_equations[random_equation_index].get_boolean_equation()}\n")
        return self._boolean_equations

    def balance_mutation(self, number_of_mutations: int):
        """
        Introduces mutations to the balance by changing link operators in the Boolean Equations.
        The method randomly selects equations from the Boolean model and mutates their link operators.

        :param number_of_mutations: The number of mutations to introduce.
        :return: The list of updated Boolean equations.
        """
        for _ in range(number_of_mutations):
            random_equation_index = random.randint(0, len(self._boolean_equations) - 1)
            orig = self._boolean_equations[random_equation_index].get_boolean_equation()
            self._boolean_equations[random_equation_index].mutate_link_operator()
            if self._boolean_equations[random_equation_index].get_boolean_equation() != orig:
                print(f"Exchanging equation {random_equation_index}\n\t{orig}\n\t"
                      f"{self._boolean_equations[random_equation_index].get_boolean_equation()}\n")
        return self._boolean_equations

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
    def file(self):
        return self._file

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

    def __str__(self):
        return f"Attractors: {self.attractors}"
