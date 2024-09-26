from typing import List, Dict

from fontTools.ttLib.tables.otBase import valueRecordFormat

from pydruglogics.utils.Util import Util
from pydruglogics.utils.Logger import Logger


class ModelOutputs:
    def __init__(self, file: str = None, model_outputs_dict: Dict[str, float] = None, verbosity=2):
        self._model_outputs: Dict[str, float] = {}
        self._logger = Logger(verbosity)
        if file is not None:
            self._load_model_outputs_file(file)
        elif model_outputs_dict is not None:
            self._load_model_outputs_dict(model_outputs_dict)
        else:
            raise ValueError('Provide either a file or a dictionary for initialization.')

        self._min_output = self._calculate_min_output()
        self._max_output = self._calculate_max_output()

    def size(self) -> int:
        return len(self._model_outputs)

    def print(self) -> None:
        try:
            print(self)
        except Exception as e:
            print(f"An error occurred while printing ModelOutputs: {e}")

    def get_model_output(self, node_name: str) -> float:
        return self._model_outputs.get(node_name, 0.0)

    def _calculate_max_output(self) -> float:
        return sum(max(weight, 0) for weight in self._model_outputs.values())

    def _calculate_min_output(self) -> float:
        return sum(min(weight, 0) for weight in self._model_outputs.values())

    def _load_model_outputs_file(self, file: str):
        self._logger.log(f"Reading model outputs file: {file}", 2)
        lines = Util.read_lines_from_file(file, True)
        for line in lines:
            node_name, weight = map(str.strip, line.split("\t"))
            self._model_outputs[node_name] = float(weight)

    def _load_model_outputs_dict(self, model_outputs_dict: Dict[str, float]):
        self._logger.log('Loading model outputs from dictionary', 2)
        self._model_outputs = model_outputs_dict

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
