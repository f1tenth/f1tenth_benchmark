import logging
import warnings
from typing import Type, Callable

import numpy as np

from f1tenth_benchmark.agent.agent import Agent
from f1tenth_benchmark.common.simulation import Simulation
from f1tenth_benchmark.task.task import Task


class Runner:
    def __init__(self, task: Task, agent: Agent, log_dir: str | None = None):
        self._task = task
        self._agent = agent
        self._log_dir = log_dir

        self._logger = logging.getLogger(__name__)

    def run(self, n_episodes: int) -> None:
        agent = self._agent

        while n_episodes > 0:
            logging.info(f"Remaining {n_episodes} episodes...")

            scene = self._task.get_next_scene()
            if scene is None:
                logging.info("No more scenes available.")
                break

            sim = Simulation()

            agent_state = agent.reset(scene=scene)
            scene.reset()

            scene_state = scene.get_state()
            sim.add(state=scene_state, action=None, agent_state=agent_state)

            while not scene.is_done():
                agent_obs = scene.observe("agent_0")
                action, agent_state = agent(
                    observation=agent_obs, agent_state=agent_state
                )

                actions = np.array([action])
                scene.step(action=actions)

                sim.add(
                    state=scene.get_state(),
                    action=action,
                    agent_state=agent_state,
                )

            metrics = self._task.monitor(sim)
            sim.add(metrics=metrics)

            sim.save(f"{self._log_dir}/episode_{n_episodes}")

            n_episodes -= 1
