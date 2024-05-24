import random
from typing import List
from gitsbe.model.Interaction import Interaction
from gitsbe.utils.Util import Util


class BooleanEquation:
    def __init__(self, arg=None):
        if arg is None:
            self.target = ''
            self.activating_regulators = {}
            self.inhibitory_regulators = {}
            self.operators_activating_regulators = []
            self.operators_inhibitory_regulators = []
            self.link = ''

        elif isinstance(arg, BooleanEquation):
            self.activating_regulators = dict(arg.activating_regulators)
            self.inhibitory_regulators = dict(arg.inhibitory_regulators)
            self.operators_activating_regulators = list(arg.operators_activating_regulators)
            self.operators_inhibitory_regulators = list(arg.operators_inhibitory_regulators)
            self.target = arg.get_target()
            self.link = arg.get_link()

        elif isinstance(arg, Interaction):
            self.activating_regulators = {}
            self.inhibitory_regulators = {}
            self.operators_activating_regulators = []
            self.operators_inhibitory_regulators = []

            tmp_activating_regulators = arg.get_activating_regulators()
            tmp_inhibitory_regulators = arg.get_inhibitory_regulators()

            self.target = arg.get_target()
            self.link = '' if len(tmp_activating_regulators) == 0 or len(tmp_inhibitory_regulators) == 0 else 'and'

            for index, regulator in enumerate(tmp_activating_regulators):
                self.activating_regulators[regulator] = 1
                if index < (len(tmp_activating_regulators) - 1):
                    self.operators_activating_regulators.append('or')

            for index, regulator in enumerate(tmp_inhibitory_regulators):
                self.inhibitory_regulators[regulator] = 1
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
            self.activating_regulators = {}
            self.inhibitory_regulators = {}
            self.operators_activating_regulators = []
            self.operators_inhibitory_regulators = []
            self.link = ''

            arg = arg.strip()
            length = len(arg)

            while True:
                arg = arg.replace('  ', ' ')
                if len(arg) == length:
                    break

            arg = arg.replace('and not', 'andnot').replace('or not', 'ornot')
            split_arg = arg.split(' ')
            self.target = split_arg[0]
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
                            self.activating_regulators[element] = 1
                        else:
                            self.inhibitory_regulators[element] = 1

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
        if len(self.activating_regulators) + len(self.inhibitory_regulators) > 1:
            is_activating = bool(random.randint(0, 1) > 0.5)
            regulators = self.activating_regulators if is_activating else self.inhibitory_regulators
            if regulators:
                random_index = random.choice(list(regulators.keys()))
                regulators[random_index] = 0 if regulators[random_index] == 1 else 1

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
            regulator_keys = list(regulators.keys())
            random.shuffle(regulator_keys)
            shuffled_regulators = {key: regulators[key] for key in regulator_keys}

            if is_activating:
                self.activating_regulators = shuffled_regulators
            else:
                self.inhibitory_regulators = shuffled_regulators

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
        activating_regulator_values = sum(self.activating_regulators.values())
        inhibitory_regulator_values = sum(self.inhibitory_regulators.values())

        if activating_regulator_values > 0:
            equation += Util.get_repeated_string(' ( ', activating_regulator_values)
            equation += ' '
            for index, (regulator, value) in enumerate(self.activating_regulators.items()):
                if value == 1:
                    if index > 0 and sum(list(self.activating_regulators.values())[:index]) > 0:
                        equation += f" {self.operators_activating_regulators[index - 1]} "
                    equation += f"{regulator} ) "

        if activating_regulator_values > 0 and inhibitory_regulator_values > 0:
            equation += self.link

        if inhibitory_regulator_values > 0:
            equation += ' not '
            equation += Util.get_repeated_string(' ( ', inhibitory_regulator_values)
            for index, (regulator, value) in enumerate(self.inhibitory_regulators.items()):
                if value == 1:
                    if index > 0 and sum(list(self.inhibitory_regulators.values())[:index]) > 0:
                        equation += f" {self.operators_inhibitory_regulators[index - 1]} "
                    equation += f"{regulator} ) "

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
        return list(self.activating_regulators.keys())

    def get_inhibitory_regulators(self) -> List[str]:
        return list(self.inhibitory_regulators.keys())

    def get_num_of_regulators(self) -> int:
        return len(self.activating_regulators) + len(self.inhibitory_regulators)

    def get_num_of_whitelisted_regulators(self) -> int:
        return (self.get_num_of_whitelisted_activating_regulators() +
                self.get_num_of_whitelisted_inhibitory_regulators())

    def get_num_of_blacklisted_regulators(self) -> int:
        return (self.get_num_of_blacklisted_activating_regulators() +
                self.get_num_of_blacklisted_inhibitory_regulators())

    def get_num_of_whitelisted_activating_regulators(self) -> int:
        return sum(value for value in self.activating_regulators.values() if value == 1)

    def get_num_of_whitelisted_inhibitory_regulators(self) -> int:
        return sum(value for value in self.inhibitory_regulators.values() if value == 1)

    def get_num_of_blacklisted_activating_regulators(self) -> int:
        return sum(value for value in self.activating_regulators.values() if value == 0)

    def get_num_of_blacklisted_inhibitory_regulators(self) -> int:
        return sum(value for value in self.inhibitory_regulators.values() if value == 0)

    def get_link(self) -> str:
        return self.link

    def set_target(self, target: str) -> None:
        self.target = target

    def set_blacklist_activating_regulators(self, index: int) -> None:
        if index >= len(self.inhibitory_regulators) or index < 0:
            raise IndexError("Index out of range")
        regulator = list(self.activating_regulators.keys())[index]
        self.activating_regulators[regulator] = 0

    def set_blacklist_inhibitory_regulators(self, index: int) -> None:
        if index >= len(self.inhibitory_regulators) or index < 0:
            raise IndexError("Index out of range")
        regulator = list(self.inhibitory_regulators.keys())[index]
        self.inhibitory_regulators[regulator] = 0
