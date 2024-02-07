from datetime import datetime

import numpy as np


class Simulation:
    def __init__(self):
        self._metadata = {
            "creation_time": datetime.now().isoformat(),
        }

        self._trajectory = {
            "states": [],
            "actions": [],
            "agent_infos": [],
            "scene_infos": [],
        }

    def add(self, state, action, agent_info, scene_info) -> None:
        self._trajectory["states"].append(state)
        self._trajectory["actions"].append(action)
        self._trajectory["agent_infos"].append(agent_info)
        self._trajectory["scene_infos"].append(scene_info)

    def save(self, path: str) -> None:
        if not path.endswith(".npz"):
            path += ".npz"
        np.savez_compressed(path, metadata=self._metadata, **self._trajectory)

    def load(self, path: str) -> None:
        assert path.endswith(".npz"), "Invalid file format"
        assert all(len(self._trajectory[key]) == 0 for key in self._trajectory), "Trajectory data already exists"

        data = np.load(path)
        assert "metadata" in data, "Invalid file format, missing metadata"
        assert all(key in data for key in self._trajectory), "Invalid file format, missing trajectory data"
        self._trajectory = {key: data[key] for key in self._trajectory}

    def render(self) -> None:
        """
        Render the recorded simulation.
        """
        raise NotImplementedError
