from typing import List, Dict
from pydruglogics.utils.BNetworkUtil import BNetworkUtil
import logging


class InteractionModel:
    def __init__(self, interactions_file=None, model_name='', remove_self_regulated_interactions=False,
                 remove_inputs=False,remove_outputs=False):
        """
        Initializes the InteractionModel from .sif file.
        :param interactions_file: File path to the .sif file with network interactions.
        :param model_name: Name for the model. An empty string by default.
        :param remove_self_regulated_interactions: If True, removes self-regulating interactions
        (source equals target).False by default.
        :param remove_inputs: If True, removes nodes without incoming interactions. False by default.
        :param remove_outputs: If True, removes nodes without outgoing interactions. False by default.
        """
        self._interactions: List[Dict] = []
        self._model_name = model_name

        if interactions_file is not None:
            self._load_sif_file(interactions_file)
            if remove_self_regulated_interactions:
                self._remove_self_regulated_interactions()
            self._remove_interactions(remove_inputs, remove_outputs)
            self._build_multiple_interactions()
        else:
            raise ValueError("An 'interactions_file' must be provided for initialization.")

    def _load_sif_file(self, interactions_file: str) -> None:
        """
        Loads all the lines of the .sif file and initializes the interactions.
        Each line is parsed to identify interaction types, which can include:
        - '<->' split into '<-' and '->'
        - '|-|' split into '|-' and '-|'
        - '|->' split into '->' and '|-'
        - '<-|' split into '<-' and '-|'
        :param interactions_file: File path to the .sif file with network interactions.
        :return None
        """
        try:
            file_extension = BNetworkUtil.get_file_extension(interactions_file)
            if file_extension != 'sif':
                print('New file extension used to load general model, currently not supported')
                raise IOError('ERROR: The extension needs to be .sif (other formats not yet supported)')

            self._model_name = BNetworkUtil.remove_extension(interactions_file)
            interaction_lines = BNetworkUtil.read_lines_from_file(interactions_file)

            for interaction in interaction_lines:
                if interaction:
                    if '<->' in interaction:
                        line1 = interaction.replace('<->', '<-')
                        line2 = interaction.replace('<->', '->')
                        self._interactions.extend(
                            [BNetworkUtil.parse_interaction(line1), BNetworkUtil.parse_interaction(line2)])
                    elif '|-|' in interaction:
                        line1 = interaction.replace('|-|', '|-')
                        line2 = interaction.replace('|-|', '-|')
                        self._interactions.extend(
                            [BNetworkUtil.parse_interaction(line1), BNetworkUtil.parse_interaction(line2)])
                    elif '|->' in interaction:
                        line1 = interaction.replace('|->', '->')
                        line2 = interaction.replace('|->', '|-')
                        self._interactions.extend(
                            [BNetworkUtil.parse_interaction(line1), BNetworkUtil.parse_interaction(line2)])
                    elif '<-|' in interaction:
                        line1 = interaction.replace('<-|', '<-')
                        line2 = interaction.replace('<-|', '-|')
                        self._interactions.extend(
                            [BNetworkUtil.parse_interaction(line1), BNetworkUtil.parse_interaction(line2)])
                    else:
                        self._interactions.append(BNetworkUtil.parse_interaction(interaction))

            logging.info('Interactions loaded successfully')

        except IOError as e:
            logging.error(f"Error reading the interactions file '{interactions_file}': {str(e)}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading the file '{interactions_file}': {str(e)}")
            raise

    def _remove_interactions(self, is_input: bool = False, is_output: bool = False) -> None:
        """
        Removes interactions based on input and output criteria.
        :param is_input: Removes interactions that are only inputs. By default,  False.
        :param is_output: Removes interactions that are only outputs. By default,  False.
        :return None
        """
        interactions_size_before_trim = len(self._interactions)
        iteration_trim = 0
        if (is_input and not is_output) or (not is_input and is_output):
            logging.debug(f"Removing ({'inputs' if is_input else 'outputs'}). "
                  f"Interactions before trim: {interactions_size_before_trim}\n")
        else:
            logging.debug(f"Interactions before trim: {interactions_size_before_trim}\n")

        while True:
            iteration_trim += 1
            interactions_size_before_trim = len(self._interactions)

            for i in range(len(self._interactions) - 1, -1, -1):
                source = self._interactions[i]['source'] if is_input else None
                target = self._interactions[i]['target'] if is_output else None

                if target and self._is_not_a_source(target):
                    logging.debug(f"Removing interaction (i = {i})  (not source):  {self._interactions[i]}")
                    self._interactions.pop(i)
                if source and self._is_not_a_target(source):
                    logging.debug(f"Removing interaction (i = {i})  (not target):  {self._interactions[i]}")
                    self._interactions.pop(i)

            if interactions_size_before_trim <= len(self._interactions):
                break
        logging.debug(f"Interactions after trim ({iteration_trim} iterations): {len(self._interactions)}\n")

    def _remove_self_regulated_interactions(self) -> None:
        """
        Removes interactions that are self regulated.
        :return None
        """
        for i in range(len(self._interactions) - 1, -1, -1):
            target = self._interactions[i]['target']
            source = self._interactions[i]['source']
            if target == source:
                logging.debug(f"Removing self regulation:  {self._interactions[i]}")
                self._interactions.pop(i)

    def _build_multiple_interactions(self) -> None:
        """
        Creates interactions with multiple regulators for every single target.
        :return None
        """
        checked_targets = {}
        multiple_interaction = []
        logging.debug('Building Boolean Equations for Interactions (.sif).')
        for interaction in self._interactions:
            target = interaction['target']
            if target not in checked_targets:
                checked_targets[target] = {'activating_regulators': set(), 'inhibitory_regulators': set()}

            match interaction['arc']:
                case 1:
                    checked_targets[target]['activating_regulators'].add(interaction['source'])
                case -1:
                    checked_targets[target]['inhibitory_regulators'].add(interaction['source'])
                case _:
                    raise RuntimeError(f"ERROR: Invalid interaction detected. Source '{interaction['source']}', "
                                       f"target '{interaction['target']}' with an unsupported value "
                                       f"'{interaction['arc']}'.")

        for target, regulators in checked_targets.items():
            new_interaction = BNetworkUtil.create_interaction(target=target)
            for activating_regulator in regulators["activating_regulators"]:
                new_interaction['activating_regulators'].append(activating_regulator)
            for inhibitory_regulator in regulators["inhibitory_regulators"]:
                new_interaction['inhibitory_regulators'].append(inhibitory_regulator)
            multiple_interaction.append(new_interaction)

        sources = {interaction['source'] for interaction in self._interactions}
        for source in sources:
            if source not in checked_targets and self._is_not_a_target(source):
                interaction = BNetworkUtil.create_interaction(target=source)
                interaction['activating_regulators'].append(source)
                multiple_interaction.append(interaction)

        self._interactions = multiple_interaction

    def size(self) -> int:
        return len(self._interactions)

    def _is_not_a_source(self, node_name: str) -> bool:
        result = True
        for interaction in self._interactions:
            if node_name == interaction['source']:
                result = False
        return result

    def _is_not_a_target(self, node_name: str) -> bool:
        result = True
        for interaction in self._interactions:
            if node_name == interaction['target']:
                result = False
        return result

    def print(self) -> None:
        try:
            print(str(self))
        except Exception as e:
            logging.error(f"An error occurred while printing the Interactions: {str(e)}")
            raise

    @property
    def interactions(self) -> List[Dict]:
        return self._interactions

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def all_targets(self) -> List[str]:
        return [interaction['target'] for interaction in self.interactions]

    def get_interaction(self, index: int) -> Dict:
        return self.interactions[index]

    def get_target(self, index: int) -> str:
        return self.interactions[index]['target']

    def get_activating_regulators(self, index: int) -> List[str]:
        return self.interactions[index]['activating_regulators']

    def get_inhibitory_regulators(self, index: int) -> List[str]:
        return self.interactions[index]['inhibitory_regulators']

    @model_name.setter
    def model_name(self, model_name: str) -> None:
        self._model_name = model_name

    def __str__(self):
        multiple_interaction = ''
        for interaction in self._interactions:
            interaction_str = f"Target: {interaction['target']}"
            if interaction['activating_regulators']:
                activators_str = ', '.join(interaction['activating_regulators'])
                interaction_str += f", activating regulators: {activators_str}"

            if interaction['inhibitory_regulators']:
                inhibitors_str = ', '.join(interaction['inhibitory_regulators'])
                interaction_str += f", inhibitory regulators: {inhibitors_str}"

            multiple_interaction += interaction_str + "\n"

        return multiple_interaction
