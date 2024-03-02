import abc
from typing import List
from .measure import Measure

class TerminationCondition(abc.ABC):
    measures: List[Measure]

    @abc.abstractmethod
    def is_done(self, obs: dict) -> bool:
        raise NotImplementedError()