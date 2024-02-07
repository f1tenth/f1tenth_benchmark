from f1tenth_benchmark.common.simulation import Simulation
from f1tenth_benchmark.monitor.monitor import Monitor


class MonitorControlTask(Monitor):
    @property
    def metrics(self) -> list[str]:
        return ["tracking_error", "completion_rate"]

    def evaluate(self, simulation: Simulation) -> dict[str, float]:
        # todo: implement eval metrics
        return {"tracking_error": 0.0, "completion_rate": 0.0}
