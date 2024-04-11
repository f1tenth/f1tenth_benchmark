import abc
from typing import List
from .measure import MultiStepMetric

class TerminationCondition(abc.ABC):
    measures: List[MultiStepMetric]

    @abc.abstractmethod
    def is_done(self, obs: dict) -> bool:
        raise NotImplementedError()