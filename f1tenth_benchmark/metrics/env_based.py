from ..core.measure import MultiStepMetric
from f1tenth_gym.envs import F110Env
from f1tenth_planning.utils.utils import nearest_point

import numpy as np

class EnvDone(MultiStepMetric):
    def __init__(self):
        super().__init__(0, "env_done")

    def evaluate(self, obs) -> bool:
        self.cumulative_score += 1 if obs["done"] else 0
        return obs["done"]

    def get_final_score(self):
        return self.cumulative_score

class DistanceToWaypoints(MultiStepMetric):
    def __init__(self, agent_id, waypoints):
        super().__init__(agent_id, "distance_to_waypoints")
        self.waypoints = waypoints
        self.cumulative_score = 0.0

    def reset(self) -> None:
        super().reset()
        self.cumulative_score = 0.0

    def evaluate(self, obs) -> float:
        distance = self._distance_to_waypoints(obs)
        self.cumulative_score += distance
        return distance

    def get_final_score(self):
        return self.cumulative_score

    def _distance_to_waypoints(self, obs) -> float:
        x, y = obs['obs']["poses_x"][self.agent_id], obs['obs']["poses_y"][self.agent_id]
        closest_pt_i = nearest_point(np.array([x, y]), self.waypoints)[-1]
        return np.linalg.norm(self.waypoints[closest_pt_i, :] - np.array([x, y]))