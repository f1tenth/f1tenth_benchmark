import collections
import logging
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from f1tenth_planning.utils.utils import nearest_point

from f1tenth_benchmark.common.simulation import Simulation
from f1tenth_benchmark.common.termination_fn import Timeout, OnAnyTermination, AnyCrossedFinishLine, DefaultTermination
from f1tenth_benchmark.scene.gym_scene import GymScene
from f1tenth_benchmark.task.task import Task
import gymnasium as gym


class TaskControl(Task):
    """
    The control task is a simple task where the agent needs to follow a predefined raceline.
    - Scenes: The scenes are defined by a sequence of poses along the raceline.
    - Monitor: The monitor evaluates the tracking error and the completion rate.
    """

    def __init__(self, render_mode: str | None = "human_fast"):
        self._config = {
            "name": self.name,
            "env": {
                "map": "Spielberg",
                "num_agents": 1,
                "control_input": "speed",
                "observation_config": {"type": "kinematic_state"},
            },
            "n_pos_track": 10,
            "timeout": 6000,
        }
        self._metrics = {}

        self._env = gym.make(
            "f110_gym:f110-v0",
            config=self._config["env"],
            render_mode=render_mode,
        )

        # create a queue of ids to starting poses along the centerline
        n_points = self._config["n_pos_track"]
        raceline = self._env.unwrapped.track.raceline
        ids = np.linspace(0, len(raceline.xs) - 1, n_points + 1, dtype=int)[:-1]
        self._poses = np.array([[raceline.xs[i], raceline.ys[i], raceline.yaws[i]] for i in ids])
        self._current_scene_id = 0

        # task termination
        self._term_fns = OnAnyTermination(fns=[
            Timeout(max_steps=6000),
            DefaultTermination(),
            AnyCrossedFinishLine()
        ])

        self._logger = logging.getLogger(__name__)



    @property
    def name(self):
        return "control"

    @property
    def config(self):
        return self._config

    def reset(self):
        self._current_scene_id = 0

    def get_next_scene(self) -> GymScene | None:
        if len(self._poses) == 0:
            return None

        next_pose = self._poses[self._current_scene_id]
        self._current_scene_id += 1

        return GymScene(self._env, options={"poses": np.array([next_pose])}, termination_fn=self._term_fns)

    @property
    def monitor(self) -> Callable[[Simulation], dict]:
        def evaluate(scene: GymScene, sim: Simulation):
            """
            Evaluate the simulation by computing the tracking error and the completion rate.
            """
            all_states = sim.get("state")

            # extract xy-trajectory
            xs = [s["agent_0"]["pose_x"] for s in all_states]
            ys = [s["agent_0"]["pose_y"] for s in all_states]
            positions = np.array([xs, ys]).T

            # extract xy-raceline
            track = scene.get_track()
            raceline_xys = np.array([track.raceline.xs, track.raceline.ys]).T

            # compute the nearest point on the raceline for each position
            nearests = [nearest_point(positions[i], raceline_xys) for i in range(len(positions))]

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
