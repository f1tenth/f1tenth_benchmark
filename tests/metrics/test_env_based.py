from f1tenth_benchmark.metrics.env_based import EnvDone, DistanceToWaypoints
import numpy as np
import pytest

def test_env_done():
    ms = EnvDone()
    assert ms.evaluate({"done": True}) == True

def test_distance_to_waypoints():
    waypoints = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    ms = DistanceToWaypoints(0, waypoints)
    pytest.approx(ms.evaluate({"obs": {"poses_x": [0.01, 10.0, 10.0], "poses_y": [0.01, 10.0, 10.0]}}), 0.0)
    pytest.approx(ms.evaluate({"obs": {"poses_x": [1.01, 11.0, 11.0], "poses_y": [1.01, 11.0, 11.0]}}), 0.0)
    pytest.approx(ms.evaluate({"obs": {"poses_x": [2.01, 12.0, 12.0], "poses_y": [2.01, 12.0, 12.0]}}), 0.0)