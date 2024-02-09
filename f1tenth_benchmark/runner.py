import logging
from tqdm import tqdm

import numpy as np

from f1tenth_benchmark.agent.agent import Agent
from f1tenth_benchmark.common.simulation import Simulation
from f1tenth_benchmark.task.task import Task


class Runner:
    def __init__(self, task: Task, agent: Agent, logger=None):
        self._task = task
        self._agent = agent
        self._logger = logger

        self._txt_logger = logging.getLogger(__name__)

    def run(self, n_episodes: int) -> dict:
        agent = self._agent
        monitor = self._task.monitor

        desclen, trunc = 35, "..."
        desc = f"Agent: {agent.name}, Task: {self._task.name}"
        desc = desc.ljust(desclen, " ")
        desc = desc[: desclen - len(trunc)] + trunc if len(desc) > desclen else desc

        all_metrics = {}
        for i in tqdm(range(n_episodes), desc=desc):
            scene = self._task.get_next_scene()
            if scene is None:
                self._txt_logger.info(
                    "No more scenes available. Stopping the benchmark."
                )
                break

            sim = Simulation()

            agent.reset(scene=scene)
            scene.reset()

            scene_state = scene.get_state()
            sim.add(state=scene_state, action=None)

            while not scene.is_done():
                agent_obs = scene.observe("agent_0")
                action = agent(observation=agent_obs)

                actions = np.array([action])
                scene.step(action=actions)

                sim.add(
                    state=scene.get_state(), action=action,
                )

            metrics = monitor(scene=scene, sim=sim)
            sim.add(metrics=metrics)

            for key, value in metrics.items():
                if key in all_metrics:
                    all_metrics[key].append(value)
                else:
                    all_metrics[key] = [value]

            # self._logger.log(sim)
            n_episodes -= 1

        return all_metrics
