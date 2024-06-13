from typing import List

import biolqm
import mpbn
import pyboolnet
from pyboolnet.trap_spaces import compute_trap_spaces, compute_steady_states
from pyboolnet import file_exchange

from gitsbe.model.ModelOutputs import ModelOutputs
from gitsbe.model.BooleanEquation import BooleanEquation
from gitsbe.utils.Util import Util


class BooleanModel:
    def __init__(self, model=None, file='', attractor_tool=''):
        self._model_name = None
        self._boolean_equations = []
        self._attractors = []
        self._attractor_tool = attractor_tool
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
    def attractors(self) -> object:
        return self._attractors

    @property
    def attractor_tool(self) -> str:
        return self._attractor_tool

    def __str__(self):
        return f"Attractors: {self.attractors}, BE: {self.boolean_equations}"
