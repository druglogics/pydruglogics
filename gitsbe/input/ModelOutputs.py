from typing import List, Dict
from gitsbe.utils.Util import Util


class ModelOutputs:
    def __init__(self, file: str):
        self._model_outputs: Dict[str, float] = {}
        self._load_model_outputs_file(file)
        self._min_output = self._calculate_min_output()
        self._max_output = self._calculate_max_output()

    def size(self) -> int:
        return len(self._model_outputs)

    def get_model_output(self, node_name: str) -> float:
        return self._model_outputs.get(node_name, 0.0)

    def _calculate_max_output(self) -> float:
        return sum(max(weight, 0) for weight in self._model_outputs.values())

    def _calculate_min_output(self) -> float:
        return sum(min(weight, 0) for weight in self._model_outputs.values())

    def _load_model_outputs_file(self, file: str):
        print(f"Reading model outputs file: {file}")
        lines = Util.read_lines_from_file(file, True)
        for line in lines:
            node_name, weight = map(str.strip, line.split("\t"))
            self._model_outputs[node_name] = float(weight)

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
        return "\n".join(f"Model output {node_name} with weight: {weight}"
                         for node_name, weight in self._model_outputs.items())
