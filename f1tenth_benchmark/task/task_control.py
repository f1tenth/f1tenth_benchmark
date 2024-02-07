from typing import Any

from f1tenth_benchmark.monitor.monitor import Monitor
from f1tenth_benchmark.monitor.monitor_control import MonitorControlTask
from f1tenth_benchmark.scene.gym_scene import GymScene
from f1tenth_benchmark.task.task import Task
import gymnasium as gym


class TaskControl(Task):
    """
    Benchmark task to assess the control capabilities of the agent.
    """

    def __init__(self):
        self._config = {}
        self._metrics = {}

        self._env = gym.make("f110_gym:f110-v0")
        self._monitor = MonitorControlTask()

    @property
    def name(self):
        return "control"

    @property
    def config(self):
        return self._config

    @property
    def metrics(self):
        return self._monitor.metrics

    def get_next_scene(self) -> Any:
        return GymScene(env=self._env)

    def get_monitor(self) -> Monitor:
        return self._monitor
