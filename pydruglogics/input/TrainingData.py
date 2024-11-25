from typing import List, Dict, Union, Tuple
from pydruglogics.utils.BNetworkUtil import BNetworkUtil
import logging

class TrainingData:
    def __init__(self, input_file: str = None, observations: List[Tuple[List[str], float]] = None):
        self._observations = []
        if input_file is not None:
            self._load_from_file(input_file)
        elif observations is not None:
            self._load_from_observations_list(observations)
        else:
            raise ValueError('Please provide a dictionary or a file.')

    def _load_from_file(self, file: str) -> None:
        try:
            lines = BNetworkUtil.read_lines_from_file(file)
        except IOError as e:
            logging.error(f"File read error: {str(e)}")
            raise

        line_index = 0
        condition, response = None, None

        while line_index < len(lines):
            line = lines[line_index].strip().lower()

            if line.startswith('condition'):
                condition = lines[line_index + 1].split("\t")
                line_index += 1
            elif line.startswith('response'):
                response = lines[line_index + 1].split("\t")
                value = response[0].split(":")[1] if 'globaloutput' in response[0] else None

                if 'globaloutput' in response[0]:
                    value = response[0].split(":")[1]

                    if not BNetworkUtil.is_numeric_string(value):
                        raise ValueError(f"Response: {response} has a non-numeric value: {value}")
                    if not (-1.0 <= float(value) <= 1.0):
                        raise ValueError(f"Response has globaloutput outside the [-1,1] range: {value}")

                line_index += 1

            elif line.startswith('weight'):
                weight = float(line.split(':')[1])

                if condition is None or response is None:
                    raise ValueError("Missing condition or response data before the weight entry.")

                self._add_observation(condition, response, weight)

            line_index += 1

        logging.info(f"Training data loaded from file: {file}.")

    def _load_from_observations_list(self, observations: List[Tuple[List[str], float]]) -> None:
        for observation in observations:
            response, weight = observation
            self._add_observation(['-'], response, weight)

        logging.info('Training data initialized from list.')

    def _add_observation(self, condition: List[str], response: List[str], weight: float) -> None:
        if 'globaloutput' in response[0]:
            value = response[0].split(":")[1]
            if not BNetworkUtil.is_numeric_string(value):
                raise ValueError(f"Response: {response} has a non-numeric value: {value}")
            if not (-1.0 <= float(value) <= 1.0):
                raise ValueError(f"Response has globaloutput outside the [-1,1] range: {value}")

        self._observations.append({
            'condition': condition,
            'response': response,
            'weight': weight
        })

    def print(self) -> None:
        try:
            print(str(self))
        except Exception as e:
            logging.error(f"Error while printing TrainingData: {str(e)}")
            raise

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
