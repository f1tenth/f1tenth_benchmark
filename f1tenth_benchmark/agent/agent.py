from abc import abstractmethod, ABC
from typing import Any

from f1tenth_benchmark.scene.scene import Scene


class Agent(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, observation):
        raise NotImplementedError()

    @abstractmethod
    def reset(self, scene: Scene) -> None:
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError()
