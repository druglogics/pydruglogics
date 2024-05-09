from typing import List


class MultipleInteraction:
    def __init__(self, target_name):
        self.target = target_name
        self.activating_regulators = []
        self.inhibitory_regulators = []
        self.activating_regulator_complexes = []
        self.inhibitory_regulator_complexes = []

    def add_activating_regulator(self, regulator: str) -> None:
        self.activating_regulators.append(regulator)

    def add_inhibitory_regulator(self, regulator: str) -> None:
        self.inhibitory_regulators.append(regulator)

    def add_activating_regulator_complexes(self, regulators: List[str]) -> None:
        self.activating_regulator_complexes.extend(regulators)

    def add_inhibitory_regulator_complexes(self, regulators: List[str]) -> None:
        self.inhibitory_regulator_complexes.extend(regulators)

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
