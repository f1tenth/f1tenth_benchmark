import abc
from .scene import Scene

class Agent(abc.ABC):

    def __init__(self, id: int) -> None:
        self.id = id

    @abc.abstractmethod
    def act(self, obs: dict) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self, scene: Scene) -> None:
        raise NotImplementedError()


class RandomAgent(Agent):
    def __init__(self, id: int) -> None:
        super().__init__(id)

    def act(self, obs: dict) -> str:
        import numpy as np
        return [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]

    def reset(self, scene: Scene) -> None:
        pass