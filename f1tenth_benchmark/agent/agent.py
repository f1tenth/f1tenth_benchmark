from abc import abstractmethod, ABC
from typing import Any

from f1tenth_benchmark.scene.scene import Scene


class Agent(ABC):

    @abstractmethod
    def __call__(self, observation, agent_state):
        raise NotImplementedError()

    @abstractmethod
    def reset(self, scene: Scene) -> Any | None:
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError()