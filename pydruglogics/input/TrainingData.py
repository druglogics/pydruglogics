from typing import List, Dict, Union, Tuple
from pydruglogics.utils.Util import Util


class TrainingData:
    def __init__(self, file: str = None, observations: List[Tuple[List[str], List[str], float]] = None):
        self._observations = []
        if file is not None:
            self._load_from_file(file)
        elif observations is not None:
            self._load_from_observations_list(observations)
        else:
            raise ValueError('Please provide a dictionary or a file.')

    def print(self) -> None:
        try:
            print(self)
        except Exception as e:
            print(f"An error occurred while printing TrainingData: {e}")

    def _load_from_file(self, file: str) -> None:
        print(f"Reading training data observations file: {file}")
        lines = Util.read_lines_from_file(file)
        line_index = 0
        while line_index < len(lines):
            line = lines[line_index].strip().lower()
            if line == 'condition':
                condition = lines[line_index + 1].split("\t")
                line_index += 1
            elif line == 'response':
                response = lines[line_index + 1].split("\t")
                if 'globaloutput' in response:
                    value = response.split(":")[1]
                    if not Util.is_numeric_string(value):
                        raise ValueError(f"Response: {response} has a non-numeric value: {value}")
                    if not (-1.0 <= float(value) <= 1.0):
                        raise ValueError(f"Response has globaloutput outside the [-1,1] range: {value}")
                line_index += 1
            elif line.startswith('weight'):
                weight = float(line.split(':')[1])
                self._observations.append({
                    'condition': condition,
                    'response': response,
                    'weight': weight
                })
            line_index += 1

    def _load_from_observations_list(self, observations: List[Tuple[List[str], List[str], float]]) -> None:
        for observation in observations:
            condition, response, weight = observation
            self._add_observation(condition, response, weight)

    def _add_observation(self, condition: List[str], response: List[str], weight: float) -> None:
        if isinstance(response, str) and 'globaloutput' in response:
            value = response.split(":")[1]
            if not Util.is_numeric_string(value):
                raise ValueError(f"Response: {response} has a non-numeric value: {value}")
            if not (-1.0 <= float(value) <= 1.0):
                raise ValueError(f"Response has globaloutput outside the [-1,1] range: {value}")

        self._observations.append({
            'condition': condition,
            'response': response,
            'weight': weight
        })

    @property
    def weight_sum(self) -> float:
        return sum(observation['weight'] for observation in self._observations)

    def size(self) -> int:
        return len(self._observations)

    @property
    def observations(self) -> List[Dict[str, Union[str, float]]]:
        return self._observations

    @property
    def responses(self) -> List[str]:
        return [item for sublist in (obs['response'] for obs in self._observations) for item in sublist]

    @property
    def response(self) -> List[str]:
        return self._observations[0]['response'] if self._observations else []

    @property
    def weights(self) -> List[float]:
        return [obs['weight'] for obs in self._observations]

    def __str__(self) -> str:
        if not self._observations:
            return "No observations available."
        observations_str = []
        for observation in self._observations:
            observations_str.append(
                f"Observation:\nCondition: {', '.join(observation['condition'])}\n"
                f"Response: {', '.join(observation['response'])}\n"
                f"Weight: {observation['weight']}\n"
            )
        return "\n".join(observations_str)
