from typing import List

from gitsbe.model.SingleInteraction import SingleInteraction
from gitsbe.model.MultipleInteraction import MultipleInteraction
from gitsbe.utils.Util import Util


class GeneralModel:
    def __init__(self, single_interactions=None):
        if single_interactions is None:
            single_interactions = []
        self.single_interactions = single_interactions
        self.multiple_interactions = []
        self.model_name = None

    def load_interactions_file(self, interactions_file: str) -> None:
        """
        check if interactions .sif file exists and call the load_sif_file method
        :param interactions_file:
        :return None:
        """
        file_extension = Util.get_file_extension(interactions_file)
        if file_extension == '.sif':
            self.load_sif_file(interactions_file)
        else:
            print('New file extension used to load general model, currently not supported')
            raise IOError('ERROR: The extension needs to be .sif (other formats not yet supported)')

    def load_sif_file(self, file_name: str) -> None:
        """
        loads all the lines of the .sif file and initialize the single_interactions
        :param file_name:
        :return None
        """
        self.model_name = Util.remove_extension(file_name)
        file_lines = Util.read_lines_from_file(file_name)
        interactions = list(file_lines)

        for interaction in interactions:
            if interaction:
                if '<->' in interaction:
                    line1 = interaction.replace('<->', '<-')
                    line2 = interaction.replace('<->', '->')
                    single_interaction1 = SingleInteraction(line1)
                    single_interaction2 = SingleInteraction(line2)
                    self.single_interactions.extend([single_interaction1, single_interaction2])
                elif '|-|' in interaction:
                    line1 = interaction.replace('|-|', '|-')
                    line2 = interaction.replace('|-|', '-|')
                    single_interaction1 = SingleInteraction(line1)
                    single_interaction2 = SingleInteraction(line2)
                    self.single_interactions.extend([single_interaction1, single_interaction2])
                elif '|->' in interaction:
                    line1 = interaction.replace('|->', '->')
                    line2 = interaction.replace('|->', '|-')
                    single_interaction1 = SingleInteraction(line1)
                    single_interaction2 = SingleInteraction(line2)
                    self.single_interactions.extend([single_interaction1, single_interaction2])
                elif '<-|' in interaction:
                    line1 = interaction.replace('<-|', '<-')
                    line2 = interaction.replace('<-|', '-|')
                    single_interaction1 = SingleInteraction(line1)
                    single_interaction2 = SingleInteraction(line2)
                    self.single_interactions.extend([single_interaction1, single_interaction2])
                else:
                    single_interaction = SingleInteraction(interaction)
                    self.single_interactions.append(single_interaction)

    def build_multiple_interactions(self) -> None:
        """
        convert list of single interactions into interactions with multiple
        regulators for every single target
        :return None
        """
        for index in range(len(self.single_interactions)):
            target = self.single_interactions[index].target
            if self.get_index_of_target_in_multiple_interactions(target) < 0:
                multiple_interaction = MultipleInteraction(target)
                for single_inter in self.single_interactions:
                    if single_inter.get_target() == target:
                        match single_inter.get_arc():
                            case 1:
                                multiple_interaction.add_activating_regulator(single_inter.get_source())
                            case -1:
                                multiple_interaction.add_inhibitory_regulator(single_inter.get_source())
                            case _:
                                raise RuntimeError('ERROR: Interaction effect malformed')
                self.multiple_interactions.append(multiple_interaction)

        for single_interact in self.single_interactions:
            probable_source_node = single_interact.get_source()
            if (self.get_index_of_target_in_multiple_interactions(probable_source_node) < 0 and
                    self.is_not_a_target(probable_source_node)):
                multiple_interact = MultipleInteraction(probable_source_node)
                multiple_interact.add_activating_regulator(probable_source_node)
                self.multiple_interactions.append(multiple_interact)

    def remove_interactions(self, is_input: bool = False, is_output: bool = False) -> None:
        """
        removes interactions from single interactions, output nodes if it has no
        outgoing edges, input nodes if it has no incoming edges or both input and output
        :param is_input:
        :param is_output:
        :return None
        """
        interactions_size_before_trim = len(self.single_interactions)
        iteration_trim = 0
        if (is_input and not is_output) or (not is_input and is_output):
            print(f"Removing ({'inputs' if is_input else 'outputs'}). "
                  f"Interactions before trim: {interactions_size_before_trim}\n")
        else:
            print(f"Interactions before trim: {interactions_size_before_trim}\n")

        while True:
            iteration_trim += 1
            interactions_size_before_trim = len(self.single_interactions)

            for i in range(len(self.single_interactions) - 1, -1, -1):
                source = self.single_interactions[i].get_source() if is_input else None
                target = self.single_interactions[i].get_target() if is_output else None

                if target and self.is_not_a_source(target):
                    print(f"Removing interaction (i = {i})  (not source):  "
                          f"{self.single_interactions[i].get_interaction()}")
                    self.single_interactions.pop(i)
                if source and self.is_not_a_target(source):
                    print(f"Removing interaction (i = {i})  (not target):  "
                          f"{self.single_interactions[i].get_interaction()}")
                    self.single_interactions.pop(i)

            if interactions_size_before_trim <= len(self.single_interactions):
                break
        print(f"Interactions after trim ({iteration_trim} iterations): {len(self.single_interactions)}\n")

    def remove_self_regulated_interactions(self) -> None:
        """
        removes interactions from single interactions that are self regulated
        :return None
        """
        for i in range(len(self.single_interactions) - 1, -1, -1):
            target = self.single_interactions[i].get_target()
            source = self.single_interactions[i].get_source()
            if target == source:
                print(f"Removing self regulation:  {self.single_interactions[i].get_interaction()}")
                self.single_interactions.pop(i)

    def size(self) -> int:
        return len(self.multiple_interactions)

    def is_not_a_source(self, node_name: str) -> bool:
        result = True
        for single_interaction in self.single_interactions:
            if node_name == single_interaction.get_source():
                result = False
        return result

    def is_not_a_target(self, node_name: str) -> bool:
        result = True
        for single_interaction in self.single_interactions:
            if node_name == single_interaction.get_target():
                result = False
        return result

    def get_model_name(self) -> str:
        return self.model_name

    def get_single_interactions(self) -> List[SingleInteraction]:
        return self.single_interactions

    def get_multiple_interactions(self) -> List[MultipleInteraction]:
        return self.multiple_interactions

    def get_multiple_interaction(self, index: int) -> MultipleInteraction:
        return self.multiple_interactions[index]

    def get_index_of_target_in_multiple_interactions(self, target: str) -> int:
        for index in range(len(self.multiple_interactions)):
            if target == self.multiple_interactions[index].get_target():
                return index
        return -1

    def set_model_name(self, model_name: str) -> None:
        self.model_name = model_name

    def __str__(self):
        return '\n'.join(map(str, self.multiple_interactions))
