from abc import ABC, abstractmethod
from typing import Iterator, Tuple
from f1tenth_gym import F110Env
from f1tenth_gym.envs.track import Track

from .termination import TerminationCondition
from .measure import MultiStepMetric


class Scene():
    def __init__(self, config) -> None:
        self.config = config
        
    def reset(self, env: F110Env) -> Tuple[dict, dict]:
        env.configure(self.config)
        obs, info = env.reset(options={'poses': self.config['poses']})
        return obs, info
    

class SceneGenerator(ABC):
    seed: int
    length: int
    
    def __len__(self) -> int:
        return self.length

    @abstractmethod
    def generate(self, config: dict) -> Scene:
        raise NotImplementedError()
    

class SimpleSceneGenerator(SceneGenerator):
    def __init__(self, seed: int, length: int) -> None:
        import numpy as np
        from f1tenth_gym.envs.reset.masked_reset import AllTrackResetFn
        self.seed = seed
        self.length = length
        self.default_config = F110Env.default_config()
        self.random_state = np.random.RandomState(seed)
        self.track = Track.from_track_name(self.default_config['map_name'])
        self.pose_generator = AllTrackResetFn(self.track.centerline, self.default_config['num_agents'])

    def generate(self, config: dict) -> Iterator[Scene]:
        for _ in range(len(self)):
            reset_poses = self.pose_generator.sample()
            config['poses'] = reset_poses
            yield Scene(config)


