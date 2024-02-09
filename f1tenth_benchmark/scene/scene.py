from abc import ABC, abstractmethod
from typing import Any

from f110_gym.envs.track import Track

Observation = Any
Action = Any


class Scene(ABC):
    @abstractmethod
    def get_track(self) -> Track:
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def observe(self, agent_id: str) -> Observation:
        raise NotImplementedError()

    @abstractmethod
    def step(self, action: Action) -> None:
        raise NotImplementedError()

    @abstractmethod
    def is_done(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_state(self, agent_id: str) -> Observation:
        raise NotImplementedError()
