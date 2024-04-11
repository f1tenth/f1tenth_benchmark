import abc
from abc import abstract, abstractmethod
from typing import Dict
from .trace_manager import TraceManager


class MultiStepMetric(abc.ABC):
    trace_manager: TraceManager
    cumulative_score: float
    historical_scores: Dict[float, float]
    agent_id: str

    @abc.abstractmethod
    def evaluate(self, obs) -> float:
        raise NotImplementedError()

    def register_trace_manager(self, trace_manager):
        self.trace_manager = trace_manager