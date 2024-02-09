from typing import Any

from f110_gym.envs.track import Track

from f1tenth_benchmark.agent.agent import Agent
from f1tenth_planning.control.pure_pursuit.pure_pursuit import PurePursuitPlanner
from f1tenth_planning.control.stanley.stanley import StanleyPlanner
from f1tenth_planning.control.lqr.lqr import LQRPlanner

from f1tenth_benchmark.scene.scene import Scene, Observation

F110_AGENT_REGISTRY = {
    "pure_pursuit": PurePursuitPlanner,
    "stanley": StanleyPlanner,
    "lqr": LQRPlanner,
}
class F110Agent(Agent):

    def __init__(self, name: str) -> None:
        self._name = name
        self._agent_fn = F110_AGENT_REGISTRY[name]
        self._agent = None

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, observation: Observation):
        assert self._agent is not None, "Agent not initialized, did you call reset?"
        action = self._agent.plan(observation)
        return action

    def reset(self, scene: Scene) -> None:
        track = scene.get_track()
        self._agent = self._agent_fn(track=track)
        return None

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass



