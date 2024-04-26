import abc
from typing import Dict, Union, Any
from .trace_manager import TraceManager


class MultiStepMetric(abc.ABC):        
    def __init__(
        self,
        agent_id: int,
        name: str,
    ) -> None: 
        self.trace_manager: Union[TraceManager, None] = None
        self.cumulative_score: float = 0.0
        self.historical_scores: Dict[float, float] = dict()
        self.agent_id = agent_id
        self.name = name
    
    def reset(self) -> None:
        self.cumulative_score = 0.0
        self.historical_scores = list()

    @abc.abstractmethod
    def evaluate(self, obs) -> Any:
        raise NotImplementedError()

    def get_final_score(self) -> Any:
        return self.cumulative_score

    def register_trace_manager(self, trace_manager):
        self.trace_manager = trace_manager