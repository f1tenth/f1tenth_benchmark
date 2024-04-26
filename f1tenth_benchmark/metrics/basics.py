from typing import Any
from ..core.measure import MultiStepMetric

class StepCounter(MultiStepMetric):
    def __init__(self):
        super().__init__(0, "step_counter")

    def evaluate(self, obs) -> int:
        self.cumulative_score += 1
        return 1

    def get_final_score(self) -> Any:
        return len(self.trace_manager)