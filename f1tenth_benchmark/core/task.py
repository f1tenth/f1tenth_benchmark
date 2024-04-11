from abc import ABC, abstractmethod
from typing import List

from f1tenth_benchmark.core.measure import MultiStepMetric
from f1tenth_benchmark.core.termination import TerminationCondition
from f1tenth_benchmark.core.trace_manager import TraceManager
from f1tenth_benchmark.core.scene import Scene

class Task(ABC):
    """
    The Task class is an abstract class that defines the interface for a task in the benchmark.
    """
    def __init__(self, 
                 termination_condition: TerminationCondition, 
                 objectives: List[MultiStepMetric], 
                 observation_config: dict, 
                 trace_manager: TraceManager) -> None:
        self.termination_condition = termination_condition
        self.objectives = objectives
        self.observation_config = observation_config
        self.trace_manager = trace_manager

    @abstractmethod
    def run(self, scene: Scene, agents: List[str]) -> dict:
        """
        Run the task on the given scene with the given agents.
        """
        raise NotImplementedError()
    
    def register_trace_manager(self, trace_manager: TraceManager):
        self.trace_manager = trace_manager
        for objective in self.objectives:
            objective.register_trace_manager(trace_manager)
        self.termination_condition.register_trace_manager(trace_manager)