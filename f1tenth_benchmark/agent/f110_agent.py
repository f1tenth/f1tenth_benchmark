from typing import Any

from f1tenth_benchmark.agent.agent import Agent
from f1tenth_planning.control.pure_pursuit.pure_pursuit import PurePursuitPlanner

F110_AGENT_REGISTRY = {
    "pure_pursuit": PurePursuitPlanner,
}
class F110Agent(Agent):

    def __init__(self, name: str) -> None:
        self.name = name
        self._agent = F110_AGENT_REGISTRY[name]()

    def __call__(self, observation, agent_state):
        return self._agent.plan(observation)

    def reset(self) -> Any | None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass



