from abc import ABC, abstractmethod
from gitsbe.model import BooleanModel


class BooleanModelOptimizer(ABC):
    @abstractmethod
    def run(self) -> [BooleanModel]:
        pass

    @abstractmethod
    def save_to_file_models(self, path):
        pass
