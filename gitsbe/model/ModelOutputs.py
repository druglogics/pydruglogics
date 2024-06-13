from typing import List, Dict
from gitsbe.utils.Util import Util


class ModelOutputs:
    _instance = None

    def __init__(self, file: str):
        if ModelOutputs._instance is not None:
            raise AssertionError('ModelOutputs instance already exists. Use get_instance() to access it.')
        self._model_outputs: List[Dict[str, float]] = []
        self._output_weights: List[Dict[str, int]] = []
        self._load_model_outputs_file(file)
        self._min_output = self._calculate_min_output()
        self._max_output = self._calculate_max_output()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise AssertionError('You have to call init first to initialize the ModelOutputs')
        return cls._instance

    @classmethod
    def initialize(cls, file: str):
        if ModelOutputs._instance is not None:
            raise AssertionError("You have already initialized me")
        ModelOutputs._instance = ModelOutputs(file)

    @classmethod
    def reset(cls):
        cls._instance = None

    def size(self) -> int:
        return len(self._model_outputs)

    def get_model_output(self, index: int) -> Dict[str, float]:
        return self._model_outputs[index]

    def _calculate_max_output(self) -> float:
        max_output = sum(max(output_weight['weight'], 0) for output_weight in self._model_outputs)
        return max_output

    def _calculate_min_output(self) -> float:
        min_output = sum(min(output_weight['weight'], 0) for output_weight in self._model_outputs)
        return min_output

    def _load_model_outputs_file(self, file: str):
        print(f"Reading model outputs file: {file}")

        lines = Util.read_lines_from_file(file, True)

        for line in lines:
            temp = line.split("\t")
            self._model_outputs.append({
                'node_name': temp[0].strip(),
                'weight': float(temp[1].strip())
            })

    @property
    def node_names(self) -> List[str]:
        return [output_weight['node_name'] for output_weight in self._model_outputs]

    @property
    def output_weights(self) -> List[Dict[str, int]]:
        return self._output_weights

    @property
    def model_outputs(self) -> List[Dict[str, float]]:
        return self._model_outputs

    @property
    def min_output(self) -> float:
        return self._min_output

    @property
    def max_output(self) -> float:
        return self._max_output

    def __str__(self):
        return [f"Modeloutput {output_weight['node_name']} with weight: {output_weight['weight']}"
                for output_weight in self._model_outputs]
