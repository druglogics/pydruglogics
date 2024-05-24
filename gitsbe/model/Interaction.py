from typing import List


class Interaction:
    def __init__(self, interaction=None, target=None, source=None):
        self.activating_regulators = []
        self.inhibitory_regulators = []
        self.activating_regulator_complexes = []
        self.inhibitory_regulator_complexes = []

        if source is None and target is None:
            tmp = interaction.split('\t')
            if len(tmp) != 3:
                raise ValueError(f"ERROR: Wrongly formatted interaction: {interaction}")
            source = tmp[0]
            interaction = tmp[1]
            target = tmp[2]

        match interaction:
            case 'activate' | 'activates' | '->':
                self.arc = 1
                self.source = source
                self.target = target
            case 'inhibit' | 'inhibits' | '-|':
                self.arc = -1
                self.source = source
                self.target = target
            case '<-':
                self.arc = 1
                self.source = source
                self.target = target
            case '|-':
                self.arc = -1
                self.source = source
                self.target = target
            case '|->' | '<->' | '<-|' | '|-|':
                print('ERROR: Wrongly formatted interaction type:')
                print(f"Source: {source} Interaction type: {interaction} Target: {target}")
                raise SystemExit(1)

    def add_activating_regulator(self, regulator: str) -> None:
        self.activating_regulators.append(regulator)

    def add_inhibitory_regulator(self, regulator: str) -> None:
        self.inhibitory_regulators.append(regulator)

    def add_activating_regulator_complexes(self, regulators: List[str]) -> None:
        self.activating_regulator_complexes.extend(regulators)

    def add_inhibitory_regulator_complexes(self, regulators: List[str]) -> None:
        self.inhibitory_regulator_complexes.extend(regulators)

    def get_source(self) -> str:
        return self.source

    def get_interaction(self) -> str:
        return f"{self.source} {'->' if self.arc == 1 else '-|'} {self.target}"

    def get_arc(self) -> int:
        return self.arc

    def get_target(self) -> str:
        return self.target

    def get_activating_regulators(self) -> List[str]:
        return self.activating_regulators

    def get_inhibitory_regulators(self) -> List[str]:
        return self.inhibitory_regulators

    def get_activating_regulator_complexes(self) -> List[str]:
        return self.activating_regulator_complexes

    def get_inhibitory_regulator_complexes(self) -> List[str]:
        return self.inhibitory_regulator_complexes

    def __str__(self):
        return (f"{self.target} <- {self.activating_regulators} {self.activating_regulator_complexes} ! "
                f"{self.inhibitory_regulators} {self.inhibitory_regulator_complexes}")
