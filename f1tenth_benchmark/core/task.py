from abc import ABC, abstractmethod
from typing import List

from f1tenth_gym.envs import F110Env
from .measure import MultiStepMetric
from .termination import TerminationCondition
from .trace_manager import TraceManager
from .scene import Scene
from .agent import Agent

import numpy as np

class Task(ABC):
    """
    The Task class is an abstract class that defines the interface for a task in the benchmark.
    """
    def __init__(self, 
                 termination_condition: TerminationCondition, 
                 objectives: List[MultiStepMetric], 
                 observation_config: dict) -> None:
        self.termination_condition = termination_condition
        self.objectives = objectives
        self.observation_config = observation_config

    def run(self, env: F110Env, scene: Scene, agents: List[Agent]) -> dict:
        """Run the task on a scene with a list of agents.

        Parameters
        ----------
        scene : Scene
            The Scene to run the task on 
        agents : List[Agent]
            A list of agents

        Returns
        -------
        dict
            A dictionary of objectives and their scores
            Example:
            {
                'objective1': 0.5,
                'objective2': 0.7,
                ...
            }
        """
        objective_history = []
        agent_ids = [agent.id for agent in agents]
        ordered_agent_ids = np.argsort(agent_ids).astype(int)
        self.trace_manager.reset()
        obs, info = scene.reset(env)
        for agent in agents:
            agent.reset(scene)
        done = False
        self.trace_manager.record_obs({
            'obs': obs,
            'info': info,
        })
        while not done:
            actions = [agents[i].act(obs) for i in ordered_agent_ids]
            packaged_actions = np.hstack(actions).reshape(-1,2)
            obs, rewards, dones, truncateds, infos = env.step(packaged_actions)
            observation_dict = {
                'obs': obs,
                'reward': rewards,
                'done': dones,
                'truncated': truncateds,
                'info': infos,
            }
            done = self.termination_condition.is_done(observation_dict)
            self.trace_manager.record_obs(observation_dict)
            objectives_dict = {objective.name: objective.evaluate(observation_dict) for objective in self.objectives}
            objective_history.append(objectives_dict)
        # We are now done, so we can compute the final objectives
        final_objectives = {objective.name: objective.get_final_score() for objective in self.objectives}
        return final_objectives
        
    
    def register_trace_manager(self, trace_manager: TraceManager):
        self.trace_manager = trace_manager
        for objective in self.objectives:
            objective.register_trace_manager(trace_manager)
        self.termination_condition.register_trace_manager(trace_manager)