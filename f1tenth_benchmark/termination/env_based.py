from ..core.termination import TerminationCondition
from ..core.measure import MultiStepMetric
from ..metrics.basics import StepCounter
from f1tenth_planning.utils.utils import nearest_point
import numpy as np
from typing import List

class CrossedFinishLine(TerminationCondition):
    def __init__(self, agent_id: str):
        raise NotImplementedError("Implement this function")
    

class TimeStepTermination(TerminationCondition):
    def __init__(self, agent_id: str, max_time_steps: int, name: str = "time_step_termination"):
        self.max_time_steps = max_time_steps
        measures = [StepCounter()]
        super().__init__(agent_id, measures, name)

    def is_done(self, obs) -> bool:
        return obs["step_count"] >= self.max_time_steps
    
class EnvTermination(TerminationCondition):
    def __init__(self, name: str = "env_termination"):
        measures = []
        self._terminated = False
        super().__init__(0, measures, name)

    def reset(self) -> None:
        self._terminated = False
        super().reset()

    def is_done(self, obs) -> bool:
        if obs["done"]:
            self._terminated = True
        return self._terminated