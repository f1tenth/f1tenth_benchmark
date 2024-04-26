from typing import List, Dict, Any, Union
from f1tenth_benchmark.core.task import Task
from f1tenth_benchmark.core.scene import Scene, SceneGenerator
from f1tenth_benchmark.core.agent import Agent
from f1tenth_gym.envs import F110Env
from f1tenth_benchmark.core.trace_manager import TraceManager


class Runner:
    def __init__(self, task: Task, scenes: Union[Scene, SceneGenerator], agents: List[Agent], env: F110Env, seed: int=42):
        self.task: Task = task
        self.scenes: Union[Scene, SceneGenerator] = scenes
        self.agents: List[Agent] = agents
        self.env: F110Env = env
        self.seed = seed
        self.trace_manager: TraceManager = TraceManager()
        self.task.register_trace_manager(self.trace_manager)
        self.results = [None for _ in range(len(self.scenes))]

    def _run_single_scene(self, scene: Scene) -> Dict[str, Any]:
        """Run the specified task on a single scene

        Parameters
        ----------
        scene : Scene
            The scene to run the task on

        Returns
        -------
        Dict["objective_name", "results"]
            Returns results for the scene
        """
        results = self.task.run(self.env, scene, self.agents)
        return results

    def run(self) -> Dict[str, Any]:
        """Run the specified task on the scenes provided

        Returns
        -------
        Dict["objective_name", "results"]
            Returns a list of results for each scene in the task
        """
        for i, scene in enumerate(self.scenes):
            results = self._run_single_scene(scene)
            self.results[i]= results
        return self.results