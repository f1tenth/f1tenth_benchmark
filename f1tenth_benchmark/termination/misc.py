from ..core.termination import TerminationCondition
from ..core.measure import MultiStepMetric


class Timeout(TerminationCondition):
    def __init__(self, max_steps: int):
        self._max_steps = max_steps
        self._steps = 0

    def reset(self) -> None:
        self._steps = 0

    def __call__(self, state, terminated, truncated, info) -> bool:
        self._steps += 1
        timeout = self._steps >= self._max_steps

        if timeout:
            self._logger.debug(f"Timeout after {self._steps} steps")

        return timeout