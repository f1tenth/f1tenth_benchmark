from typing import Any, Dict, List
from functools import lru_cache

class TraceManager:
    """This class maintains a copy of the trace of the environment in a centralized manner.
    It is used to compute metrics that require the full trace of the environment, or some subset of it.
    """
    historic_trace: List[Dict[str, Any]]

    def __init__(self):
        self.historic_trace = []

    def __len__(self):
        return len(self.historic_trace)
    
    def reset(self):
        self.historic_trace = []
    
    def record_obs(self, obs: Dict[str, Any]):
        self.historic_trace.append(obs)

    @property
    @lru_cache(maxsize=128)
    def trace(self):
        return self.historic_trace