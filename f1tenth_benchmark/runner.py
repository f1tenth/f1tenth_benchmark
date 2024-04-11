from tqdm import tqdm

from typing import List, Union
from f1tenth_benchmark.core.agent import Agent
from f1tenth_benchmark.core.scene import Scene, SceneGenerator
from f1tenth_benchmark.core.task import Task
from f1tenth_benchmark.core.trace_manager import TraceManager
from f1tenth_gym.envs import F110Env

class Runner:
    def __init__(
        self,
        task: Task,
        scenes: Union[SceneGenerator, List[Scene]],
        agents: List[Agent],
        env: F110Env,
        trace_manager: TraceManager,
        ):
        self.task = task
        self.scenes = scenes
        self.agents = agents
        self.env = env
        self.trace_manager = trace_manager
    
    def run(self):
        for scene in tqdm(self.scenes):
            self.trace_manager.reset()
            objectives = self.task.run(scene, self.agents)