from f1tenth_benchmark.task.task import Task


class TaskControl(Task):
    """
    Benchmark task to assess the control capabilities of the agent.
    """

    def __init__(self):
        self._config = {}
        self._metrics = {}

    @property
    def name(self):
        return "control"

    @property
    def config(self):
        return self._config

    @property
    def metrics(self):
        return self._metrics
