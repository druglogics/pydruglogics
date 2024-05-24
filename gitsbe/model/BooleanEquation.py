import random
from typing import List

from gitsbe.model.Interaction import Interaction
from gitsbe.utils.Util import Util


class BooleanEquation:
    def __init__(self, arg=None):
        if arg is None:
            self.target = ''
            self.activating_regulators = []
            self.inhibitory_regulators = []
            self.operators_activating_regulators = []
            self.operators_inhibitory_regulators = []
            self.whitelist_activating_regulators = []
            self.whitelist_inhibitory_regulators = []
            self.link = ''

        elif isinstance(arg, BooleanEquation):
            self.activating_regulators = []
            self.inhibitory_regulators = []
            self.operators_activating_regulators = []
            self.operators_inhibitory_regulators = []
            self.whitelist_activating_regulators = []
            self.whitelist_inhibitory_regulators = []

            self.target = arg.get_target()
            self.link = arg.get_link()
            self.operators_activating_regulators.extend(arg.operators_activating_regulators)
            self.operators_inhibitory_regulators.extend(arg.operators_inhibitory_regulators)
            for index, _ in enumerate(arg.activating_regulators):
                self.activating_regulators.append(arg.activating_regulators[index])
                self.whitelist_activating_regulators.append(arg.whitelist_activating_regulators[index])
            for index, _ in enumerate(arg.inhibitory_regulators):
                self.inhibitory_regulators.append(arg.inhibitory_regulators[index])
                self.whitelist_inhibitory_regulators.append(arg.whitelist_inhibitory_regulators[index])

        elif isinstance(arg, Interaction):
            self.activating_regulators = []
            self.inhibitory_regulators = []
            self.operators_activating_regulators = []
            self.operators_inhibitory_regulators = []
            self.whitelist_activating_regulators = []
            self.whitelist_inhibitory_regulators = []

            tmp_activating_regulators = arg.get_activating_regulators()
            tmp_inhibitory_regulators = arg.get_inhibitory_regulators()

            self.target = arg.get_target()
            self.link = '' if len(tmp_activating_regulators) == 0 or len(tmp_inhibitory_regulators) == 0 else 'and'

            for index, _ in enumerate(tmp_activating_regulators):
                self.activating_regulators.append(tmp_activating_regulators[index])
                self.whitelist_activating_regulators.append(True)
                if index < (len(tmp_activating_regulators) - 1):
                    self.operators_activating_regulators.append('or')

            for index, _ in enumerate(tmp_inhibitory_regulators):
                self.inhibitory_regulators.append(tmp_inhibitory_regulators[index])
                self.whitelist_inhibitory_regulators.append(True)
                if index < (len(tmp_inhibitory_regulators) - 1):
                    self.operators_inhibitory_regulators.append('or')

        elif isinstance(arg, str):
            """
            Build equation from Boolean expression. Currently expressions must be of the
            following type (the <i>booleannet</i> format - <i>and</i> can be exchanged with <i>or</i>)
            <b>A *= ( ( ( B ) or C ) or D ) and not ( ( ( E ) or F ) or G )</b>
            <br>
            Spaces between parentheses and node names are essential (the parentheses themselves not so much)
            """
            self.activating_regulators = []
            self.inhibitory_regulators = []
            self.operators_activating_regulators = []
            self.operators_inhibitory_regulators = []
            self.whitelist_activating_regulators = []
            self.whitelist_inhibitory_regulators = []

            arg = arg.strip()
            length = len(arg)

            while True:
                arg = arg.replace('  ', ' ')
                if len(arg) == length:
                    break

            arg = arg.replace('and not', 'andnot')
            arg = arg.replace('or not', 'ornot')

            split_arg = arg.split(' ')
            self.target = split_arg[0]
            self.link = ''
            split_arg.pop(0)
            before_not = True

            while split_arg:
                element = split_arg[0].strip()
                split_arg.pop(0)

                match element:
                    case '*=' | '(' | ')':
                        pass
                    case 'andnot':
                        before_not = False
                        self.link = 'and'
                    case 'ornot':
                        before_not = False
                        self.link = 'or'
                    case 'not':
                        before_not = False
                    case 'or' | 'and':
                        if before_not:
                            self.operators_activating_regulators.append(element)
                        else:
                            self.operators_inhibitory_regulators.append(element)
                    case _:
                        if before_not:
                            self.activating_regulators.append(element)
                            self.whitelist_activating_regulators.append(True)
                        else:
                            self.inhibitory_regulators.append(element)
                            self.whitelist_inhibitory_regulators.append(True)

    def mutate_random_operator(self) -> None:
        """
        Randomly select an acitvation or inhibiting operator and mutate it.
        :return: None
        """
        is_activating = bool(random.randint(0, 1) > 0.5)
        operators = self.operators_activating_regulators if is_activating else self.operators_inhibitory_regulators
        if operators:
            random_index = random.randint(0, len(operators) - 1)
            if operators[random_index].strip() == 'or':
                operators[random_index] = 'and'
            else:
                operators[random_index] = 'or'

    def mutate_regulator(self) -> None:
        """
        Randomly select an activating or inhibiting regulator and mutate it.
        :return: None
        """
        if (self.whitelist_activating_regulators.count(True) + self.whitelist_inhibitory_regulators.count(True)) > 1:
            is_activating = bool(random.randint(0, 1) > 0.5)
            regulators = self.activating_regulators if is_activating else self.inhibitory_regulators
            whitelists = self.whitelist_activating_regulators if is_activating \
                else self.whitelist_inhibitory_regulators
            if regulators:
                random_index = random.randint(0, len(regulators) - 1)
                whitelists[random_index] = not whitelists[random_index]

    def mutate_link_operator(self) -> None:
        self.link = 'or' if self.link.strip() == 'and' else 'and'

    def shuffle_random_regulatory_priority(self) -> None:
        """
        Randomly shuffle the priorities of the regulators.
        e.g.
        (  (  A )  or B ) or not  (  (  ( A )  or B )  or C )
        (  (  B )  or A ) or not  (  (  ( B )  or A )  or C )
        :return: None
        """
        is_activating = bool(random.randint(0, 1) == 1)
        regulators = self.activating_regulators if is_activating else self.inhibitory_regulators
        if regulators:
            random_index = random.randint(0, len(regulators) - 2)
            tmp = regulators[random_index]
            regulators[random_index] = regulators[random_index + 1]
            regulators[random_index + 1] = tmp

    def convert_to_sif_lines(self, delimiter: str) -> List[str]:
        lines = []
        for activating_regulator in self.activating_regulators:
            lines.append(f"{activating_regulator}{delimiter}->{delimiter}{self.target}")
        for inhibitory_regulator in self.inhibitory_regulators:
            lines.append(f"{inhibitory_regulator}{delimiter}-|{delimiter}{self.target}")
        return lines

    def get_boolean_equation(self) -> str:
        """
        Returns the string of the equation represented in the Booleannet format: <br>
        <i>A *=  (  (  B )  or C or ...) and not  (  ( E )  or F or ...)</i>
        :return: str
        """
        equation = f"{self.target} *= "

        if self.whitelist_activating_regulators.count(True) > 0:
            equation += Util.get_repeated_string(' ( ', self.whitelist_activating_regulators.count(True))
            equation += ' '

            for index, _ in enumerate(self.activating_regulators):
                if self.whitelist_activating_regulators[index]:
                    if self.whitelist_activating_regulators[:index].count(True) > 0:
                        equation += f" {self.operators_activating_regulators[index - 1]} "
                    equation += f"{self.activating_regulators[index]} ) "

        if (self.whitelist_activating_regulators.count(True) > 0 and
                self.whitelist_inhibitory_regulators.count(True) > 0):
            equation += self.link

        if self.whitelist_inhibitory_regulators.count(True) > 0:
            equation += ' not '
            equation += Util.get_repeated_string(' ( ', self.whitelist_inhibitory_regulators.count(True))
            for index in range(len(self.inhibitory_regulators)):
                if self.whitelist_inhibitory_regulators[:index].count(True):
                    equation += f" {self.operators_inhibitory_regulators[index - 1]} "
                equation += f"{self.activating_regulators[index]} ) "

        return f" {equation.strip()} "

    def get_interactions(self) -> List[Interaction]:
        interactions = []
        for activating_regulator in self.activating_regulators:
            interactions.append(Interaction('->', activating_regulator, self.target))
        for inhibitory_regulator in self.inhibitory_regulators:
            interactions.append(Interaction('-|', inhibitory_regulator, self.target))
        return interactions

    def get_target(self) -> str:
        return self.target

    def get_activating_regulators(self) -> List[str]:
        return self.activating_regulators

    def get_inhibitory_regulators(self) -> List[str]:
        return self.inhibitory_regulators

    def get_num_of_regulators(self) -> int:
        return len(self.activating_regulators) + len(self.inhibitory_regulators)

    def get_num_of_whitelisted_regulators(self) -> int:
        return (self.get_num_of_whitelisted_activating_regulators() +
                self.get_num_of_whitelisted_inhibitory_regulators())

    def get_num_of_blacklisted_regulators(self) -> int:
        return (self.get_num_of_blacklisted_activating_regulators() +
                self.get_num_of_blacklisted_inhibitory_regulators())

    def get_num_of_whitelisted_activating_regulators(self) -> int:
        return self.whitelist_activating_regulators.count(True)

    def get_num_of_whitelisted_inhibitory_regulators(self) -> int:
        return self.whitelist_inhibitory_regulators.count(True)

    def get_num_of_blacklisted_activating_regulators(self) -> int:
        return self.whitelist_activating_regulators.count(False)

    def get_num_of_blacklisted_inhibitory_regulators(self) -> int:
        return self.whitelist_inhibitory_regulators.count(False)

    def get_link(self) -> str:
        return self.link

    def set_target(self, target: str) -> None:
        self.target = target

    def set_blacklist_activating_regulators(self, index: int) -> None:
        if index >= len(self.inhibitory_regulators) or index < 0:
            raise IndexError("Index out of range")
        self.whitelist_activating_regulators[index] = False

    def set_blacklist_inhibitory_regulators(self, index: int) -> None:
        if index >= len(self.inhibitory_regulators) or index < 0:
            raise IndexError("Index out of range")
        self.whitelist_inhibitory_regulators[index] = False
