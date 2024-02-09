import logging
from abc import abstractmethod
from typing import Callable

import numpy as np
from f110_gym.envs.track import Track
from f1tenth_planning.utils.utils import nearest_point

from f1tenth_benchmark.scene.scene import Scene


class TerminationFn(Callable):
    @abstractmethod
    def reset(self, scene: Scene) -> None:
        pass

    @abstractmethod
    def __call__(self, state, terminated, truncated, info) -> bool:
        raise NotImplementedError


class OnAnyTermination(TerminationFn):
    def __init__(self, fns: list[TerminationFn]) -> None:
        self._fns = fns

    def reset(self, scene: Scene) -> None:
        for fn in self._fns:
            fn.reset(scene=scene)

    def __call__(self, state, terminated, truncated, info) -> bool:
        return any(fn(state, terminated, truncated, info) for fn in self._fns)


class OnAllTermination(TerminationFn):
    def __init__(self, fns: list[TerminationFn]) -> None:
        self._fns = fns

    def reset(self, scene: Scene) -> None:
        for fn in self._fns:
            fn.reset(scene=scene)

    def __call__(self, state, terminated, truncated, info) -> bool:
        return all(fn(state, terminated, truncated, info) for fn in self._fns)


class Timeout(TerminationFn):
    def __init__(self, max_steps: int) -> None:
        self._max_steps = max_steps
        self._steps = 0

        self._logger = logging.getLogger(__name__)

    def reset(self, scene: Scene) -> None:
        self._steps = 0

    def __call__(self, state, terminated, truncated, info) -> bool:
        self._steps += 1
        timeout = self._steps >= self._max_steps

        if timeout:
            self._logger.debug(f"Timeout after {self._steps} steps")

        return timeout


class AnyCrossedFinishLine(TerminationFn):
    def __init__(self):
        self._crossed = False
        self._state = None

        self._logger = logging.getLogger(__name__)

    def reset(self, scene: Scene) -> None:
        self._crossed = False

        track = scene.get_track()
        self._raceline_xys = np.stack([track.raceline.xs, track.raceline.ys], axis=-1)
        self._raceline_ss = track.raceline.ss

        self._prev_ss = None

    def __call__(self, state, terminated, truncated, info) -> bool:
        # first get the s value from agents' positions
        new_ss = np.zeros(len(state))
        for i, agent_id in enumerate(state):
            x, y = state[agent_id]["pose_x"], state[agent_id]["pose_y"]
            closest_pt_i = nearest_point(np.array([x, y]), self._raceline_xys)[-1]
            new_ss[i] = self._raceline_ss[closest_pt_i]

        # check if any agent crossed the finish line (in the last meter of the track)
        half_finishline = 0.5  # meters
        if self._prev_ss is not None:
            crossing = (self._prev_ss > self._raceline_ss[-1] - 2 * half_finishline) & (
                new_ss < self._raceline_ss[-1] - half_finishline
            )
            self._crossed = any(crossing)

        self._prev_ss = new_ss

        if self._crossed:
            self._logger.debug(f"Agent crossed the finish line")

        return self._crossed


class DefaultTermination(TerminationFn):
    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def reset(self, scene: Scene) -> None:
        pass

    def __call__(self, state, terminated, truncated, info) -> bool:
        done = terminated or truncated

        if done:
            self._logger.debug(
                f"Default termination: terminated={terminated}, truncated={truncated}"
            )

        return done
