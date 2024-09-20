from abc import ABC, abstractmethod
from pydruglogics.model import BooleanModel


class BooleanModelOptimizer(ABC):
    @abstractmethod
    def run(self) -> [BooleanModel]:
        pass

    @abstractmethod
    def save_to_file_models(self, path):
        pass