from abc import ABC, abstractmethod
from typing import Any

Observation = Any
Action = Any

class Scene(ABC):

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def observe(self) -> Observation:
        raise NotImplementedError()

    @abstractmethod
    def step(self, action: Action) -> None:
        raise NotImplementedError()

    @abstractmethod
    def is_done(self) -> bool:
        raise NotImplementedError()

