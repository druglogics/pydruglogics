import random
from typing import List, Dict
from gitsbe.model.InteractionModel import InteractionModel


class BooleanEquation:
    def __init__(self, arg=None, interaction_index=None):
        if arg is None:
            self._target = ''
            self._activating_regulators: Dict[str, int] = {}
            self._inhibitory_regulators: Dict[str, int] = {}
            self._operators_activating_regulators: List[str] = []
            self._operators_inhibitory_regulators: List[str] = []
            self._link = ''

        elif isinstance(arg, BooleanEquation):
            self._activating_regulators = arg.activating_regulators.copy()
            self._inhibitory_regulators = arg.inhibitory_regulators.copy()
            self._operators_activating_regulators = arg.operators_activating_regulators.copy()
            self._operators_inhibitory_regulators = arg.operators_inhibitory_regulators.copy()
            self._target = arg.target
            self._link = arg.link

        elif isinstance(arg, InteractionModel) and interaction_index is not None:
            self._activating_regulators = {}
            self._inhibitory_regulators = {}
            self._operators_activating_regulators = []
            self._operators_inhibitory_regulators = []

            self._target = arg.get_target(interaction_index)
            tmp_activating_regulators = arg.get_activating_regulators(interaction_index)
            tmp_inhibitory_regulators = arg.get_inhibitory_regulators(interaction_index)

            self._link = '' if not tmp_activating_regulators or not tmp_inhibitory_regulators else 'and'

            for i, regulator in enumerate(tmp_activating_regulators):
                self._activating_regulators[regulator] = 1
                if i < (len(tmp_activating_regulators) - 1):
                    self._operators_activating_regulators.append('or')

            for i, regulator in enumerate(tmp_inhibitory_regulators):
                self._inhibitory_regulators[regulator] = 1
                if i < (len(tmp_inhibitory_regulators) - 1):
                    self._operators_inhibitory_regulators.append('or')

        elif isinstance(arg, str):
            """
            Build equation from Boolean expression. Currently expressions must be of the
            following type (the <i>booleannet</i> format - <i>and</i> can be exchanged with <i>or</i>)
            <b>A *= ( ( ( B ) or C ) or D ) and not ( ( ( E ) or F ) or G )</b>
            <br>
            Spaces between parentheses and node names are essential (the parentheses themselves not so much)
            """
            self._activating_regulators = {}
            self._inhibitory_regulators = {}
            self._operators_activating_regulators = []
            self._operators_inhibitory_regulators = []
            self._link = ''

            arg = (arg.strip().replace('!', ' not ').replace('&', ' and ')
                   .replace('|', ' or '))
            split_arg = arg.split()
            self._target = split_arg.pop(0)
            if split_arg.pop(0) != '*=':
                raise ValueError("Equation must start with '*='")

            before_not = True

            for regulator in split_arg:
                if regulator == 'not':
                    before_not = not before_not
                elif regulator in ('and', 'or'):
                    if before_not:
                        self._operators_activating_regulators.append(regulator)
                        self._link = 'and'
                    else:
                        self._operators_inhibitory_regulators.append(regulator)
                        self._link = 'or'
                elif regulator in ('(', ')'):
                    continue
                else:
                    if before_not:
                        self._activating_regulators[regulator] = 1
                    else:
                        self._inhibitory_regulators[regulator] = 1
                        before_not = True

            self._link = '' if not self._activating_regulators or not self._inhibitory_regulators else 'and'

    def mutate_random_operator(self) -> None:
        """
        Randomly select an activating or inhibiting operator and mutate it.
        :return: None
        """
        is_activating = bool(random.randint(0, 1) > 0.5)
        operators = self._operators_activating_regulators if is_activating else self._operators_inhibitory_regulators
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
        if len(self._activating_regulators) + len(self._inhibitory_regulators) > 1:
            is_activating = bool(random.randint(0, 1) > 0.5)
            regulators = self._activating_regulators if is_activating else self._inhibitory_regulators
            if regulators:
                random_key = random.choice(list(regulators.keys()))
                regulators[random_key] = 0 if regulators[random_key] == 1 else 1
                # print('test')

    def mutate_link_operator(self) -> None:
        if self._link == '':
            pass
        else:
            self._link = 'or' if self._link.strip() == 'and' else 'and'

    def convert_to_sif_lines(self, delimiter: str) -> List[str]:
        lines = [f"{regulator}{delimiter}->{delimiter}{self._target}" for regulator in self._activating_regulators]
        lines += [f"{regulator}{delimiter}-|{delimiter}{self._target}" for regulator in self._inhibitory_regulators]
        return lines

    def get_boolean_equation(self) -> str:
        """
        Returns the string of the equation represented in the Booleannet format: <br>
        <i>A *=  B or C or ... and not E or F or ...</i>
        :return: str
        """
        equation = f"{self._target} *= "
        activating_regulator_values = sum(self._activating_regulators.values())
        inhibitory_regulator_values = sum(self._inhibitory_regulators.values())

        if activating_regulator_values > 0:
            for index, (regulator, value) in enumerate(self._activating_regulators.items()):
                if value == 1:
                    if index > 0 and sum(list(self._activating_regulators.values())[:index]) > 0:
                        equation += f" {self._operators_activating_regulators[index - 1]} "
                    equation += f"{regulator} "

        if activating_regulator_values > 0 and inhibitory_regulator_values > 0:
            equation += self._link

        if inhibitory_regulator_values > 0:
            equation += ' not '
            for index, (regulator, value) in enumerate(self._inhibitory_regulators.items()):
                if value == 1:
                    if index > 0 and sum(list(self._inhibitory_regulators.values())[:index]) > 0:
                        equation += f" {self._operators_inhibitory_regulators[index - 1]} "
                    equation += f"{regulator} "

        final_value = equation.strip()
        return final_value

    def to_bnet_format(self):
        equation = f"{self._target}, "
        activating_regulator_values = sum(self._activating_regulators.values())
        inhibitory_regulator_values = sum(self._inhibitory_regulators.values())

        activation_terms = []
        inhibition_terms = []

        if activating_regulator_values > 0:
            for index, (regulator, value) in enumerate(self._activating_regulators.items()):
                if value == 1:
                    activation_terms.append(regulator)

        if inhibitory_regulator_values > 0:
            for index, (regulator, value) in enumerate(self._inhibitory_regulators.items()):
                if value == 1:
                    inhibition_terms.append(f"!{regulator}")

        activation_expression = " & ".join(activation_terms)
        inhibition_expression = " & ".join(inhibition_terms)

        if activation_expression and inhibition_expression:
            combined_expression = f"{activation_expression} & ({inhibition_expression})"
        elif activation_expression:
            combined_expression = activation_expression
        elif inhibition_expression:
            combined_expression = inhibition_expression
        else:
            combined_expression = "false"

        final_value = f"{equation.strip()} {combined_expression.strip()}\n"

        modified_string = final_value.replace('(', '').replace(')', '')

        return modified_string

    def modify_activating_values_from_list(self, new_values):
        new_values = [int(value) for value in new_values]
        keys_act = list(self._activating_regulators.keys())
        if len(new_values) > len(keys_act):
            new_values = new_values[:len(keys_act)]
        elif len(new_values) < len(keys_act):
            extra_keys_needed = len(keys_act) - len(new_values)
            new_values.extend([0] * extra_keys_needed)

        for i, key in enumerate(keys_act):
            self._activating_regulators[key] = new_values[i]

        if len(new_values) > len(keys_act):
            for value in new_values[len(keys_act):]:
                self._activating_regulators[''] = value

    def modify_inhibitory_values_from_list(self, new_values: List[int]):
        new_values = [int(value) for value in new_values]
        keys = list(self._inhibitory_regulators.keys())
        if len(new_values) > len(keys):
            new_values = new_values[:len(keys)]
        elif len(new_values) < len(keys):
            extra_keys_needed = len(keys) - len(new_values)
            new_values.extend([0] * extra_keys_needed)

        for i, key in enumerate(keys):
            self._inhibitory_regulators[key] = new_values[i]

        if len(new_values) > len(keys):
            for value in new_values[len(keys):]:
                self._inhibitory_regulators[''] = value

    def modify_link_from_list(self, new_values: List[int]):
        new_values = [int(value) for value in new_values]
        last_value = new_values[-1]
        if last_value == -1:
            pass
        if last_value == 0:
            self._link = 'or'
        if last_value == 1:
            self._link = 'and'

    def initialize_inhibitory_regulators(self, new_values: List[float]):
        int_values = [int(value) for value in new_values[:3]]
        int_values += [0] * (3 - len(int_values))

        keys = ['key1', 'key2', 'key3']
        self._inhibitory_regulators = {keys[i]: int_values[i] for i in range(3)}

    @property
    def target(self) -> str:
        return self._target

    @target.setter
    def target(self, target: str) -> None:
        self._target = target

    @property
    def activating_regulators(self) -> Dict[str, int]:
        return self._activating_regulators

    def get_values_activating_regulators(self):
        return list(self._activating_regulators.values())

    def get_values_inhibitory_regulators(self):
        return list(self._inhibitory_regulators.values())

    @activating_regulators.setter
    def activating_regulators(self, activating_regulators: Dict[str, int]) -> None:
        self._activating_regulators = activating_regulators

    @property
    def inhibitory_regulators(self) -> Dict[str, int]:
        return self._inhibitory_regulators

    @inhibitory_regulators.setter
    def inhibitory_regulators(self, inhibitory_regulators: Dict[str, int]) -> None:
        self._inhibitory_regulators = inhibitory_regulators

    @property
    def operators_activating_regulators(self) -> List[str]:
        return self._operators_activating_regulators

    @operators_activating_regulators.setter
    def operators_activating_regulators(self, operators_activating_regulators: List[str]) -> None:
        self._operators_activating_regulators = operators_activating_regulators

    @property
    def operators_inhibitory_regulators(self) -> List[str]:
        return self._operators_inhibitory_regulators

    @operators_inhibitory_regulators.setter
    def operators_inhibitory_regulators(self, operators_inhibitory_regulators: List[str]) -> None:
        self._operators_inhibitory_regulators = operators_inhibitory_regulators

    @property
    def link(self) -> str:
        return self._link

    @link.setter
    def link(self, link: str) -> None:
        self._link = link

    def __str__(self):
        return (f"BooleanEquation: target = {self.target}, "
                f"link_operator = {self.link}, "
                f"activating_regulators = {self.activating_regulators}, "
                f"inhibitory_regulators = {self.inhibitory_regulators}, "
                f"operators_activating = {self.operators_activating_regulators}, "
                f"operators_inhibitory = {self.operators_inhibitory_regulators}")
