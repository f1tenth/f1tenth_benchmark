from typing import List, Dict, Any
from .trace_manager import TraceManager
from .agent import Agent
from .scene import Scene


class Runner:
    def __init__(self, task, scenes, agents, env, seed=42):
        self.task = task
        self.scenes = scenes
        self.agents = agents
        self.env = env
        self.seed = seed
        self.trace_manager = TraceManager()
        self.results = [None for _ in range(len(self.scenes))]

    def run(self) -> Dict[str, Any]:
        for scene in self.scenes:
            scene.config.update({'observation_config': self.task.observation_config})
            scene.reset(self.env)
            self.task.register_trace_manager(self.trace_manager)
            self.trace_manager.reset()
            obs = self.env.reset()
            self.trace_manager.record_obs(obs)
            done = False
            while not done:
                actions = {agent.name: agent.act(obs) for agent in self.agents}
                obs, _, done, _ = self.env.step(actions)
                self.trace_manager.record_obs(obs)
                done = self.task.termination_condition.is_done(obs)
                for measure in self.task.objectives:
                    measure.evaluate(obs)
            # Store the results of the task in a dictionary
            result_dict = {
                "trace": self.trace_manager.trace,
                "measures": {measure.name: measure.value for measure in self.task.objectives},
            }
            self.results.append(result_dict)
        return self.results