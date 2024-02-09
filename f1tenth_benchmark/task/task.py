from abc import ABC, abstractmethod
from typing import Any, Callable

from f1tenth_benchmark.common.simulation import Simulation
from f1tenth_benchmark.scene.scene import Scene


class Task(ABC):
    """
    A task defines the benchmarking problem to be solved by the agent,
    provides the environment scenes and monitors the agent's performance.
    """

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def config(self):
        raise NotImplementedError()

    @abstractmethod
    def get_next_scene(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()

    @property
    @abstractmethod
    def monitor(self) -> Callable[[Scene, Simulation], dict]:
        raise NotImplementedError()
