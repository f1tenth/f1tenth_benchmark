import gymnasium as gym
from f110_gym.envs.track import Track

from f1tenth_benchmark.scene.scene import Scene, Observation, Action


class GymScene(Scene):

    def __init__(self, env: gym.Env, options: dict, termination_fn: callable) -> None:
        self._env = env
        self._options = options
        self._termination_fn = termination_fn

        self._state = None
        self._info = None
        self._is_done = False

    def get_track(self) -> Track:
        return self._env.unwrapped.track

    def reset(self) -> None:
        self._is_done = False
        self._termination_fn.reset(scene=self)
        self._state, self._info = self._env.reset(options=self._options)

    def observe(self, agent_id: str) -> Observation:
        return self._state[agent_id]

    def get_state(self) -> Observation:
        return self._state

    def step(self, action: Action) -> None:
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._env.render()
        self._state = obs
        self._is_done = self._termination_fn(obs, terminated, truncated, info)
        self._info = info

    def is_done(self) -> bool:
        return self._is_done
