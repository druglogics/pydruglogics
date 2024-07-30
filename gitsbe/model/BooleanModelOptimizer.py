from abc import ABC, abstractmethod

class BooleanModelOptimizer(ABC):
    @abstractmethod
    def run(self):
        pass
