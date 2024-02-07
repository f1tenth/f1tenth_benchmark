import gymnasium as gym

from f1tenth_benchmark.scene.scene import Scene, Observation, Action


class GymScene(Scene):
    def __init__(self, env: gym.Env) -> None:
        self.env = env

        self._state = None
        self._info = None
        self._is_done = False

    def reset(self) -> None:
        self._is_done = False
        self._state, self._info = self.env.reset()

    def observe(self) -> Observation:
        return self._state

    def step(self, action: Action) -> None:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._state = obs
        self._is_done = terminated or truncated
        self._info = info

    def is_done(self) -> bool:
        return self._is_done
