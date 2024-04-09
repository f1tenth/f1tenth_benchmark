import abc
from .scene import Scene

class Agent(abc.ABC):
    @abc.abstractmethod
    def act(self, obs: dict) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self, scene: Scene) -> None:
        raise NotImplementedError()