import abc
from typing import List
from .measure import MultiStepMetric

class TerminationCondition(MultiStepMetric):
    def __init__(self, agent_id: int, measures: List[MultiStepMetric], name: str) -> None:
        self.measures: List[MultiStepMetric] = measures
        super().__init__(agent_id, name)

    @abc.abstractmethod
    def is_done(self, obs: dict) -> bool:
        raise NotImplementedError()
    
    def evaluate(self, obs: dict) -> None:
        for measure in self.measures:
            measure.evaluate(obs)

class AnyCompoundTerminationCondition(TerminationCondition):
    def __init__(self, agent_id: int, termination_conditions: List[TerminationCondition]) -> None:
        super().__init__(agent_id, termination_conditions)

    def is_done(self, obs: dict) -> bool:
        return any([measure.is_done(obs) for measure in self.measures])
    
class AllCompoundTerminationCondition(TerminationCondition):
    def __init__(self, agent_id: int, termination_conditions: List[TerminationCondition]) -> None:
        super().__init__(agent_id, termination_conditions)

    def is_done(self, obs: dict) -> bool:
        return all([measure.is_done(obs) for measure in self.measures])
    
class MetricGreaterThan(TerminationCondition):
    def __init__(self, agent_id: int, metric: MultiStepMetric, threshold: float) -> None:
        self.metric = metric
        self.threshold = threshold
        super().__init__(agent_id, [metric])

    def is_done(self, obs: dict) -> bool:
        return self.metric.get_final_score() > self.threshold