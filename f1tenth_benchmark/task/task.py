from abc import ABC, abstractmethod
from typing import Any

from f1tenth_benchmark.monitor.monitor import Monitor


class Task(ABC):
    """
    A task for the benchmark to execute.
    It consists of a task name, configuration and evaluation metrics.

    For example, the "control" task consists of a task name "control",
    configuration that includes the environment description, observations, actions, etc. and
    evaluation metrics that include the tracking error and completion rate.
    """

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def config(self):
        raise NotImplementedError()

    @property
    def metrics(self):
        raise NotImplementedError()

    @abstractmethod
    def get_next_scene(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def get_monitor(self) -> Monitor:
        raise NotImplementedError()
