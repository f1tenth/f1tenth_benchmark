from .core.agent import Agent
from .core.scene import Scene

from f1tenth_gym.envs import F110Env

import numpy as np

class RandomAgent(Agent):
    def __init__(self, id: int, env:F110Env) -> None:
        super().__init__(id)
        self.action_space = env.action_space

    def act(self, obs: dict) -> np.array:
        return self.action_space.sample()[self.id]

    def reset(self, scene: Scene) -> None:
        pass