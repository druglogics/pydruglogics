from abc import ABC, abstractmethod

class BooleanModelOptimizer(ABC):
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def save_to_file_responses(self):
        pass
