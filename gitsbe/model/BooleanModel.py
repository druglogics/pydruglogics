import random
import biolqm
import mpbn
import pyboolnet
from pyboolnet.trap_spaces import compute_trap_spaces
from pyboolnet import file_exchange
from gitsbe.input.TrainingData import TrainingData
from gitsbe.input.ModelOutputs import ModelOutputs
from gitsbe.model.BooleanEquation import BooleanEquation
from gitsbe.utils.Util import Util


class BooleanModel:
    def __init__(self, model=None, file='', attractor_tool='', model_name=''):
        self._model_name = model_name
        self._boolean_equations = []
        self._attractors = []
        self._attractor_tool = attractor_tool
        self._fitness = 0
        self._file = file

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
        self._model_name = model.model_name
        interactions = model.interactions

        print(f"Initializing BooleanModel from InteractionModel with {len(interactions)} interactions.")

        for i, interaction in enumerate(interactions):
            try:
                equation = BooleanEquation(interaction, i)
                self._boolean_equations.append(equation)
            except Exception as e:
                print(f"Error initializing BooleanEquation for interaction {interaction}: {e}")

    def init_from_file(self, file: str, attractor_tool: str) -> None:
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

        for _, line in enumerate(lines[1:], 1):
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

        self.calculate_attractors(attractor_tool)

    def calculate_attractors(self, attractor_tool: str) -> None:
        """
        calculates the attractors of the boolean model. The tool for the calculation
        is based on the value of 'self.attractor_tool'.
        Values for 'self.attractor_tool' (please choose one):
        'biolqm_trapspaces', 'biolqm_stable_states', 'mpbn_trapspaces', 'pyboolnet_trapspaces'

        :param attractor_tool:
        """
        if 'biolqm' in attractor_tool:
            self.calculate_attractors_biolqm(self._file, attractor_tool)
        elif 'mpbn' in attractor_tool:
            self.calculate_attractors_mpbn(self._file)
        else:
            self.calculate_attractors_pyboolnet(self._file)

    def calculate_attractors_biolqm(self, file, attractor_tool) -> str:
        lqm = biolqm.load(file)
        if 'stable' in attractor_tool:
            fps = biolqm.fixpoints(lqm)
            self._attractors = fps
            if self._attractors:
                return f"BioLQM found {len(self._attractors)} stable states."
        elif 'trapsace' in attractor_tool:
            fps = biolqm.trapspace(lqm)
            self._attractors = fps
            if self._attractors:
                return f"MPBN found {len(self._attractors)} trap spaces."

        return 'BioLQM found no attractors.'

    def calculate_attractors_mpbn(self, file: str) -> str:
        bn = mpbn.MPBooleanNetwork.load(file)
        attractors = bn.attractors()
        self._attractors = list(attractors)

        if self._attractors:
            return f"MPBN found {len(self._attractors)} trap spaces."

        return 'MPBN found no trap spaces.'

    def calculate_attractors_pyboolnet(self, file: str) -> str:
        primes = file_exchange.bnet_file2primes(file)
        trap_spaces = pyboolnet.trap_spaces.compute_trap_spaces(primes)
        self._attractors = trap_spaces
        if self._attractors:
            return f"PyBoolNet found {len(self._attractors)} trap spaces."

        return 'PyBoolnet found no trap spaces.'

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

    def calculate_fitness(self, attractor_tool: str) -> None:
        """
        Calculate fitness of model by going through all the observations defined in the
        training data and computing individual fitness values for each one of them.
        :param attractor_tool:
        """
        data = TrainingData.get_instance()
        self._fitness = 0

        for condition_number, observation in enumerate(data.observations):
            condition_fitness = 0
            response = observation['response']
            weight = observation['weight']
            mutated_boolean_model = BooleanModel(file=self._file)
            mutated_boolean_model.model_name = f"{self._model_name}_condition_{condition_number}"
            mutated_boolean_model.calculate_attractors(attractor_tool)
            attractors_with_nodes = mutated_boolean_model.attractors[0]

            if mutated_boolean_model.has_attractors():
                if response[0].split(":")[0] == "globaloutput":
                    observed_global_output = float(response[0].split(":")[1])
                    predicted_global_output = mutated_boolean_model.calculate_global_output()
                    condition_fitness = 1 - abs(predicted_global_output - observed_global_output)
                else:
                    if mutated_boolean_model.has_stable_states():
                        condition_fitness += 1

                    average_match = 0
                    found_observations = 0
                    matches = []

                    for index_state in range(1, attractors_with_nodes.__sizeof__()):
                        match_sum = 0
                        found_observations = 0

                        for response_str in response:
                            node, observation = response_str.split(":")
                            index_node = self.get_index_of_equation(node)

                            if index_node >= 0:
                                found_observations += 1
                                node_state = observation
                                state_value = 0.5 if node_state == "-" else float(node_state)
                                match = 1 - abs(state_value - float(observation))
                                match_sum += match

                        matches.append(match_sum)

                    for match in matches:
                        average_match += match
                    average_match /= len(matches)
                    condition_fitness += average_match

                    if found_observations > 0:
                        if mutated_boolean_model.has_stable_states():
                            condition_fitness /= (found_observations + 1)
                        else:
                            condition_fitness /= found_observations

            self._fitness += condition_fitness * weight / data.get_weight_sum()
        print(f"\nCalculating fitness..")
        print(f"Scaled fitness [0..1] for model [{self.model_name}]:  {self._fitness}")

    def get_index_of_equation(self, node_name: str) -> int:
        for index, equation in enumerate(self._boolean_equations):
            if equation.target == node_name:
                return index
        return -1

    def topology_mutations(self, number_of_mutations: int) -> None:
        """
        Introduce mutations to topology, removing regulators of nodes (not all regulators for any node)
        :param number_of_mutations:
        """
        for _ in range(number_of_mutations):
            random_equation_index = random.randint(0, len(self._boolean_equations) - 1)
            orig = self._boolean_equations[random_equation_index].get_boolean_equation()
            self._boolean_equations[random_equation_index].mutate_regulator()
            if self._boolean_equations[random_equation_index].get_boolean_equation() != orig:
                print(f"Exchanging equation {random_equation_index}\n\t{orig}\n\t"
                      f"{self._boolean_equations[random_equation_index].get_boolean_equation()}\n")

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
    def attractor_tool(self) -> str:
        return self._attractor_tool

    @model_name.setter
    def model_name(self, model_name: str) -> None:
        self._model_name = model_name

    def __str__(self):
        return f"Attractors: {self.attractors}, BE: {self.boolean_equations}"
