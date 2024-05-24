from typing import List

from gitsbe.model.Interaction import Interaction
from gitsbe.utils.Util import Util


class GeneralModel:
    def __init__(self, interactions=None):
        if interactions is None:
            interactions = []
        self.interactions = interactions
        self.model_name = None

    def load_sif_file(self, interactions_file: str) -> None:
        """
        loads all the lines of the .sif file and initialize the single_interactions
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
                    interaction1 = Interaction(line1)
                    interaction2 = Interaction(line2)
                    self.interactions.extend([interaction1, interaction2])
                elif '|-|' in interaction:
                    line1 = interaction.replace('|-|', '|-')
                    line2 = interaction.replace('|-|', '-|')
                    interaction1 = Interaction(line1)
                    interaction2 = Interaction(line2)
                    self.interactions.extend([interaction1, interaction2])
                elif '|->' in interaction:
                    line1 = interaction.replace('|->', '->')
                    line2 = interaction.replace('|->', '|-')
                    interaction1 = Interaction(line1)
                    interaction2 = Interaction(line2)
                    self.interactions.extend([interaction1, interaction2])
                elif '<-|' in interaction:
                    line1 = interaction.replace('<-|', '<-')
                    line2 = interaction.replace('<-|', '-|')
                    interaction1 = Interaction(line1)
                    interaction2 = Interaction(line2)
                    self.interactions.extend([interaction1, interaction2])
                else:
                    interaction1 = Interaction(interaction)
                    self.interactions.append(interaction1)

    def remove_interactions(self, is_input: bool = False, is_output: bool = False) -> None:
        """
        removes interactions from single interactions, output nodes if it has no
        outgoing edges, input nodes if it has no incoming edges or both input and output
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
                source = self.interactions[i].get_source() if is_input else None
                target = self.interactions[i].get_target() if is_output else None

                if target and self.is_not_a_source(target):
                    print(f"Removing interaction (i = {i})  (not source):  "
                          f"{self.interactions[i].get_interaction()}")
                    self.interactions.pop(i)
                if source and self.is_not_a_target(source):
                    print(f"Removing interaction (i = {i})  (not target):  "
                          f"{self.interactions[i].get_interaction()}")
                    self.interactions.pop(i)

            if interactions_size_before_trim <= len(self.interactions):
                break
        print(f"Interactions after trim ({iteration_trim} iterations): {len(self.interactions)}\n")

    def remove_self_regulated_interactions(self) -> None:
        """
        removes interactions from single interactions that are self regulated
        :return None
        """
        for i in range(len(self.interactions) - 1, -1, -1):
            target = self.interactions[i].get_target()
            source = self.interactions[i].get_source()
            if target == source:
                print(f"Removing self regulation:  {self.interactions[i].get_interaction()}")
                self.interactions.pop(i)

    def build_multiple_interactions(self) -> None:
        """
        creates interactions with multiple regulators for every single target
        :return None
        """
        checked_targets = {}
        multiple_interaction = []

        for interaction in self.interactions:
            target = interaction.get_target()
            if target not in checked_targets:
                checked_targets[target] = {'activating_regulators': set(), 'inhibitory_regulators': set()}

            match interaction.get_arc():
                case 1:
                    checked_targets[target]['activating_regulators'].add(interaction.get_source())
                case -1:
                    checked_targets[target]['inhibitory_regulators'].add(interaction.get_source())
                case _:
                    raise RuntimeError('ERROR: Interaction effect malformed')

        for target, regulators in checked_targets.items():
            new_interaction = Interaction(target=target, source='None', interaction='->')
            for activating_regulator in regulators["activating_regulators"]:
                new_interaction.add_activating_regulator(activating_regulator)
            for inhibitory_regulator in regulators["inhibitory_regulators"]:
                new_interaction.add_inhibitory_regulator(inhibitory_regulator)
            multiple_interaction.append(new_interaction)

        sources = {interaction.get_source() for interaction in self.interactions}
        for source in sources:
            if source not in checked_targets and self.is_not_a_target(source):
                interaction = Interaction(target=source, source='None', interaction='->')
                interaction.add_activating_regulator(source)
                multiple_interaction.append(interaction)

        self.interactions = multiple_interaction

    def size(self) -> int:
        return len(self.interactions)

    def is_not_a_source(self, node_name: str) -> bool:
        result = True
        for interaction in self.interactions:
            if node_name == interaction.get_source():
                result = False
        return result

    def is_not_a_target(self, node_name: str) -> bool:
        result = True
        for interaction in self.interactions:
            if node_name == interaction.get_target():
                result = False
        return result

    def get_interactions(self) -> List[Interaction]:
        return self.interactions

    def get_interaction(self, index: int) -> Interaction:
        return self.interactions[index]

    def get_model_name(self) -> str:
        return self.model_name

    def set_model_name(self, model_name: str) -> None:
        self.model_name = model_name

    def __str__(self):
        return '\n'.join(map(str, self.interactions))
