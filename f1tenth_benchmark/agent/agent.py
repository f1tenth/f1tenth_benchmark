from abc import abstractmethod, ABC
from typing import Any


class Agent(ABC):

    @abstractmethod
    def __call__(self, observation, agent_state):
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> Any | None:
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError()