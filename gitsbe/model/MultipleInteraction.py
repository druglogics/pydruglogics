class MultipleInteraction:
    def __init__(self, target_name):
        self.target = target_name
        self.activating_regulators = []
        self.inhibitory_regulators = []
        self.activating_regulator_complexes = []
        self.inhibitory_regulator_complexes = []

    def add_activating_regulator(self, regulator):
        self.activating_regulators.append(regulator)

    def add_inhibitory_regulator(self, regulator):
        self.inhibitory_regulators.append(regulator)

    def add_activating_regulator_complexes(self, regulators):
        self.activating_regulator_complexes.extend(regulators)

    def add_inhibitory_regulator_complexes(self, regulators):
        self.inhibitory_regulator_complexes.extend(regulators)

    def get_target(self):
        return self.target

    def get_activating_regulators(self):
        return self.activating_regulators

    def get_inhibitory_regulators(self):
        return self.inhibitory_regulators

    def get_activating_regulator_complexes(self):
        return self.activating_regulator_complexes

    def get_inhibitory_regulator_complexes(self):
        return self.inhibitory_regulator_complexes

    def __str__(self):
        return (f"{self.target} <- {self.activating_regulators} {self.activating_regulator_complexes} ! "
                f"{self.inhibitory_regulators} {self.inhibitory_regulator_complexes}")
