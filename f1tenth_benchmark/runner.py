import logging

import numpy as np

from f1tenth_benchmark.agent.agent import Agent
from f1tenth_benchmark.common.simulation import Simulation
from f1tenth_benchmark.task.task import Task


class Runner:
    def __init__(self, task: Task, agent: Agent, log_dir: str | None = None):
        self._task = task
        self._agent = agent
        self._log_dir = log_dir

    def run(self, n_episodes: int) -> None:
        monitor = self._task.get_monitor()

        while n_episodes > 0:
            logging.info(f"Remaining {n_episodes} episodes...")

            scene = self._task.get_next_scene()
            sim = Simulation()

            state = scene.reset()
            agent_state = self._agent.reset()

            sim.add(state=state, action=None, agent_info=agent_state, scene_info=None)

            while not scene.is_done():
                action, agent_info = self._agent(
                    observation=state, agent_state=agent_state
                )
                state, scene_info = scene.step(action)
                sim.add(
                    state=state,
                    action=action,
                    agent_info=agent_info,
                    scene_info=scene_info,
                )

            metrics = monitor.evaluate(sim)

            if self._log_dir is not None:
                sim.save(f"episode_{n_episodes}_sim.npz")
                np.savez_compressed(
                    f"{self._log_dir}/episode_{n_episodes}_metrics.npz", **metrics
                )

            n_episodes -= 1
