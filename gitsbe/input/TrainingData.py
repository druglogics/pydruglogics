from gitsbe.utils.Util import Util


class TrainingData:
    training_data = None

    def __init__(self, filename):
        self._observations = []
        self.load_from_file(filename)

    @classmethod
    def initialize(cls, filename):
        if cls.training_data is not None:
            raise AssertionError('The TrainingData class has already been initialized')
        cls.training_data = TrainingData(filename)

    @classmethod
    def get_instance(cls):
        if cls.training_data is None:
            raise AssertionError('You have to call init first to initialize the TrainingData class')
        return cls.training_data

    def load_from_file(self, file):
        print(f"Reading training data observations file: {file}")
        lines = Util.read_lines_from_file(file)

        condition = []
        response = []

        line_index = 0
        while line_index < len(lines):
            line = lines[line_index].lower()
            if line == 'condition':
                condition = lines[line_index + 1].split("\t")
                line_index += 1

            if line == 'response':
                response = lines[line_index + 1].split("\t")
                if 'globaloutput' in response[0]:
                    value = response[0].split(":")[1]
                    if not Util.is_numeric_string(value):
                        raise ValueError(f"Response: {response[0]} has a non-numeric value: {value}")

                    if not (-1.0 <= float(value) <= 1.0):
                        raise ValueError(f"Response has globaloutput outside "
                                         f"the [-1,1] range: {str(value)}")

                line_index += 1

            if line.startswith('weight'):
                weight = float(lines[line_index].split(':')[1])
                self._observations.append({
                    'condition': condition,
                    'response': response,
                    'weight': weight
                })
            line_index += 1
    @property
    def get_weight_sum(self):
        return sum(observation['weight'] for observation in self._observations)

    def size(self):
        return len(self._observations)

    @property
    def observations(self):
        return self._observations

    def __str__(self):
        result = []
        for i, observation in enumerate(self._observations, 1):
            result.extend([
                f"Observation {i}:",
                "Condition: " + ", ".join(observation['condition']),
                "Response: " + ", ".join(observation['response']),
                "Weight: " + str(observation['weight']),
                ""])
        return str(result)
