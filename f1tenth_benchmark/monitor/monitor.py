from abc import ABC, abstractmethod

from f1tenth_benchmark.common.simulation import Simulation


class Monitor(ABC):
    @abstractmethod
    def evaluate(self, simulation: Simulation) -> dict[str, float]:
        raise NotImplementedError
