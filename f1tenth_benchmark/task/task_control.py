import collections
import logging
from typing import Callable

import numpy as np
from f1tenth_planning.utils.utils import nearest_point

from f1tenth_benchmark.common.simulation import Simulation
from f1tenth_benchmark.common.termination_fn import Timeout, OnAnyTermination, AnyCrossedFinishLine, DefaultTermination
from f1tenth_benchmark.scene.gym_scene import GymScene
from f1tenth_benchmark.task.task import Task
import gymnasium as gym


class TaskControl(Task):
    """
    Benchmark task to assess the control capabilities of the agent.
    """

    def __init__(self, render_mode: str = "human_fast"):
        self._config = {
            "env": {
                "map": "Spielberg",
                "num_agents": 1,
                "control_input": "speed",
                "observation_config": {"type": "kinematic_state"},
            },
            "n_pos_track": 5,
        }
        self._metrics = {}

        self._env = gym.make(
            "f110_gym:f110-v0",
            config=self._config["env"],
            render_mode=render_mode,
        )

        # create a queue of ids to starting poses along the centerline
        n_points = self._config["n_pos_track"]
        raceline = self._env.track.raceline
        ids = np.linspace(0, len(raceline.xs) - 1, n_points, dtype=int)
        self._poses = collections.deque([[raceline.xs[i], raceline.ys[i], raceline.yaws[i]] for i in ids])

        self._logger = logging.getLogger(__name__)

    @property
    def name(self):
        return "control"

    @property
    def config(self):
        return self._config

    @property
    def metrics(self):
        return self._monitor.metrics

    def get_next_scene(self) -> GymScene | None:
        if len(self._poses) == 0:
            return None

        next_pose = self._poses.popleft()
        term_fn = OnAnyTermination(fns=[
            Timeout(max_steps=6000),
            DefaultTermination(),
            AnyCrossedFinishLine()
        ])
        return GymScene(self._env, options={"poses": np.array([next_pose])}, termination_fn=term_fn)

    @property
    def monitor(self) -> Callable[[Simulation], dict]:
        def evaluate(sim: Simulation):
            # todo: this is quite slow for now, need to optimize
            state = sim.get("state")
            xs = [s["agent_0"]["pose_x"] for s in state]
            ys = [s["agent_0"]["pose_y"] for s in state]
            poss = np.array([xs, ys]).T

            track = self._env.track
            raceline_xys = np.array([track.raceline.xs, track.raceline.ys]).T
            nearests = [nearest_point(poss[i], raceline_xys) for i in range(len(poss))]

            nearest_ids = np.array([near[3] for near in nearests])
            dist_to_them = [near[1] for near in nearests]

            ss = track.raceline.ss[nearest_ids]
            progress = np.sum(ss[1:] - ss[:-1])

            metrics = {
                "min_track_error": np.min(dist_to_them),
                "max_track_error": np.max(dist_to_them),
                "mean_track_error": np.mean(dist_to_them),
                "completion_rate": progress / track.raceline.ss[-1],
            }

            self._logger.debug("\n\t" + "\n\t".join([f"{k}: {v:.5f}" for k, v in metrics.items()]))

            return metrics

        return evaluate
