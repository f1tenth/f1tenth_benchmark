from f1tenth_benchmark.runner import Runner
from f1tenth_benchmark.core.task import Task
from f1tenth_benchmark.agents import RandomAgent
from f1tenth_benchmark.termination.env_based import EnvTermination
from f1tenth_benchmark.metrics.env_based import DistanceToWaypoints
from f1tenth_benchmark.core.scene import SimpleSceneGenerator

from f1tenth_gym.envs import F110Env

import numpy as np

def test_integration():
    env = F110Env()

    trial_task = Task(
        termination_condition=EnvTermination(),
        objectives=[DistanceToWaypoints(
            0, 
            np.stack([env.track.centerline.xs, env.track.centerline.ys], axis=1)
            )],
        observation_config=None,
    )

    scene_gen = SimpleSceneGenerator(seed=0, length=10)
    
    runner = Runner(
        task=trial_task,
        scenes=scene_gen,
        agents=[RandomAgent(0, env), RandomAgent(1, env)],
        env=env,
    )

    results = runner.run()
    print(results)

