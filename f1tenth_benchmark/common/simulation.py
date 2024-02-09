import logging
import warnings
from datetime import datetime
from typing import Any

from f1tenth_benchmark.common.utils import get_git_info


class Simulation:
    def __init__(self):
        self._metadata = {
            "creation_time": datetime.now().isoformat(),
            **get_git_info(),
        }

        self._data = {}

    def add(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key in self._data:
                self._data[key].append(value)
            else:
                self._data[key] = [value]


    def get(self, key: str) -> Any:
            return self._data[key]

    def save(self, path: str) -> None:
        logging.info(f"Saving simulation data to {path}...")
        warnings.warn("Saving simulation data is not yet implemented")


    def load(self, path: str) -> None:
        warnings.warn("Loading simulation data is not yet implemented")

    def render(self) -> None:
        warnings.warn("Rendering simulation data is not yet implemented")
