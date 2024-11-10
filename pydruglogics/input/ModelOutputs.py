import logging
from typing import List, Dict
from pydruglogics.utils.BNetworkUtil import BNetworkUtil


class ModelOutputs:
    def __init__(self, input_file: str = None, input_dictionary: Dict[str, float] = None):
        self._model_outputs: Dict[str, float] = {}
        if input_file is not None:
            self._load_model_outputs_file(input_file)
        elif input_dictionary is not None:
            self._load_model_outputs_dict(input_dictionary)
        else:
            raise ValueError('Provide either a file or a dictionary for initialization.')

        self._min_output = self._calculate_min_output()
        self._max_output = self._calculate_max_output()


    def _calculate_max_output(self) -> float:
        return sum(max(weight, 0) for weight in self._model_outputs.values())

    def _calculate_min_output(self) -> float:
        return sum(min(weight, 0) for weight in self._model_outputs.values())

    def _load_model_outputs_file(self, file: str):
        try:
            lines = BNetworkUtil.read_lines_from_file(file, True)
            for line in lines:
                node_name, weight = map(str.strip, line.split("\t"))
                self._model_outputs[node_name] = float(weight)

            logging.info(f"Model outputs loaded from file: {file}.")
        except IOError as e:
            logging.error(f"File read error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error while loading model outputs from file: {str(e)}")
            raise

    def _load_model_outputs_dict(self, model_outputs_dict: Dict[str, float]):
        if not isinstance(model_outputs_dict, dict):
            raise TypeError("model_outputs_dict must be a dictionary [text, number].")

        if not model_outputs_dict:
            logging.warning("model_outputs_dict is empty. No model outputs loaded.")

        self._model_outputs = model_outputs_dict
        logging.info("Model outputs are initialized from dictionary.")

    def get_model_output(self, node_name: str) -> float:
        return self._model_outputs.get(node_name, 0.0)

    def size(self) -> int:
        return len(self._model_outputs)

    def print(self) -> None:
        try:
            print(str(self))
        except Exception as e:
            logging.error(f"An error occurred while printing ModelOutputs: {str(e)}")
            raise

    @property
    def node_names(self) -> List[str]:
        return list(self._model_outputs.keys())

    @property
    def model_outputs(self) -> Dict[str, float]:
        return self._model_outputs

    @property
    def min_output(self) -> float:
        return self._min_output

    @property
    def max_output(self) -> float:
        return self._max_output

    def __str__(self) -> str:
        return "\n".join(f"Model output: {node_name}, weight: {weight}"
                         for node_name, weight in self._model_outputs.items())
