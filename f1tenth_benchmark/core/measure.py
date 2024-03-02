import abc
from abc import abstract, abstractmethod
from typing import Dict

class Measure(abc.ABC):
    cumulative_score: float
    historical_scores: Dict[float, float]
    agent_id: str

    @abc.abstractmethod
    def evaluate(self, obs) -> float:
        raise NotImplementedError()


class MultiStepMetric(Measure):
    trace_manager: TraceManager

    def register_trace_manager(self, trace_manager):
        self.trace_manager = trace_manager