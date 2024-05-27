from typing import List
from gitsbe.utils.Util import Util


class Interaction:
    def __init__(self, interactions=None):
        if interactions is None:
            interactions = []
        self.interactions = interactions
        self.model_name = None

    def load_sif_file(self, interactions_file: str) -> None:
        """
        loads all the lines of the .sif file and initializes the interactions
        :param interactions_file:
        :return None
        """
        file_extension = Util.get_file_extension(interactions_file)
        if file_extension != 'sif':
            print('New file extension used to load general model, currently not supported')
            raise IOError('ERROR: The extension needs to be .sif (other formats not yet supported)')

        self.model_name = Util.remove_extension(interactions_file)
        file_lines = Util.read_lines_from_file(interactions_file)
        interactions = list(file_lines)

        for interaction in interactions:
            if interaction:
                if '<->' in interaction:
                    line1 = interaction.replace('<->', '<-')
                    line2 = interaction.replace('<->', '->')
                    self.interactions.extend([Util.parse_interaction(line1), Util.parse_interaction(line2)])
                elif '|-|' in interaction:
                    line1 = interaction.replace('|-|', '|-')
                    line2 = interaction.replace('|-|', '-|')
                    self.interactions.extend([Util.parse_interaction(line1), Util.parse_interaction(line2)])
                elif '|->' in interaction:
                    line1 = interaction.replace('|->', '->')
                    line2 = interaction.replace('|->', '|-')
                    self.interactions.extend([Util.parse_interaction(line1), Util.parse_interaction(line2)])
                elif '<-|' in interaction:
                    line1 = interaction.replace('<-|', '<-')
                    line2 = interaction.replace('<-|', '-|')
                    self.interactions.extend([Util.parse_interaction(line1), Util.parse_interaction(line2)])
                else:
                    self.interactions.append(Util.parse_interaction(interaction))

    def remove_interactions(self, is_input: bool = False, is_output: bool = False) -> None:
        """
        removes interactions based on input and output criteria.
        :param is_input:
        :param is_output:
        :return None
        """
        interactions_size_before_trim = len(self.interactions)
        iteration_trim = 0
        if (is_input and not is_output) or (not is_input and is_output):
            print(f"Removing ({'inputs' if is_input else 'outputs'}). "
                  f"Interactions before trim: {interactions_size_before_trim}\n")
        else:
            print(f"Interactions before trim: {interactions_size_before_trim}\n")

        while True:
            iteration_trim += 1
            interactions_size_before_trim = len(self.interactions)

            for i in range(len(self.interactions) - 1, -1, -1):
                source = self.interactions[i]['source'] if is_input else None
                target = self.interactions[i]['target'] if is_output else None

                if target and self.is_not_a_source(target):
                    print(f"Removing interaction (i = {i})  (not source):  {self.interactions[i]}")
                    self.interactions.pop(i)
                if source and self.is_not_a_target(source):
                    print(f"Removing interaction (i = {i})  (not target):  {self.interactions[i]}")
                    self.interactions.pop(i)

            if interactions_size_before_trim <= len(self.interactions):
                break
        print(f"Interactions after trim ({iteration_trim} iterations): {len(self.interactions)}\n")

    def remove_self_regulated_interactions(self) -> None:
        """
        removes interactions that are self regulated
        :return None
        """
        for i in range(len(self.interactions) - 1, -1, -1):
            target = self.interactions[i]['target']
            source = self.interactions[i]['source']
            if target == source:
                print(f"Removing self regulation:  {self.interactions[i]}")
                self.interactions.pop(i)

    def build_multiple_interactions(self) -> None:
        """
        creates interactions with multiple regulators for every single target
        :return None
        """
        checked_targets = {}
        multiple_interaction = []

        for interaction in self.interactions:
            target = interaction['target']
            if target not in checked_targets:
                checked_targets[target] = {'activating_regulators': set(), 'inhibitory_regulators': set()}

            match interaction['arc']:
                case 1:
                    checked_targets[target]['activating_regulators'].add(interaction['source'])
                case -1:
                    checked_targets[target]['inhibitory_regulators'].add(interaction['source'])
                case _:
                    raise RuntimeError('ERROR: Interaction effect malformed')

        for target, regulators in checked_targets.items():
            new_interaction = Util.create_interaction(target=target)
            for activating_regulator in regulators["activating_regulators"]:
                new_interaction['activating_regulators'].append(activating_regulator)
            for inhibitory_regulator in regulators["inhibitory_regulators"]:
                new_interaction['inhibitory_regulators'].append(inhibitory_regulator)
            multiple_interaction.append(new_interaction)

        sources = {interaction['source'] for interaction in self.interactions}
        for source in sources:
            if source not in checked_targets and self.is_not_a_target(source):
                interaction = Util.create_interaction(target=source)
                interaction['activating_regulators'].append(source)
                multiple_interaction.append(interaction)

        self.interactions = multiple_interaction

    def size(self) -> int:
        return len(self.interactions)

    def is_not_a_source(self, node_name: str) -> bool:
        result = True
        for interaction in self.interactions:
            if node_name == interaction['source']:
                result = False
        return result

    def is_not_a_target(self, node_name: str) -> bool:
        result = True
        for interaction in self.interactions:
            if node_name == interaction['target']:
                result = False
        return result

    def get_interactions(self) -> List[dict]:
        return self.interactions

    def get_interaction(self, index: int) -> dict:
        return self.interactions[index]

    def get_model_name(self) -> str:
        return self.model_name

    def get_target(self, index: int) -> str:
        return self.interactions[index]['target']

    def get_all_targets(self) -> List[str]:
        return [interaction['target'] for interaction in self.interactions]

    def get_activating_regulators(self, index: int) -> List[str]:
        return self.interactions[index]['activating_regulators']

    def get_inhibitory_regulators(self, index: int) -> List[str]:
        return self.interactions[index]['inhibitory_regulators']

    def set_model_name(self, model_name: str) -> None:
        self.model_name = model_name

    def __str__(self):
        multiple_interaction = ''
        for interation in self.interactions:
            multiple_interaction += (f"{interation['target']} <- {interation['activating_regulators']} "
                                     f"{interation['activating_regulator_complexes']} ! "
                                     f"{interation['inhibitory_regulators']} "
                                     f"{interation['inhibitory_regulator_complexes']}\n")
        return multiple_interaction
