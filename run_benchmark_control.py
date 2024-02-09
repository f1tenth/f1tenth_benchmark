import logging

import numpy as np

from f1tenth_benchmark.agent.f110_agent import F110Agent
from f1tenth_benchmark.runner import Runner
from f1tenth_benchmark.task.task_control import TaskControl

import plotly.graph_objects as go


def plot_metrics_bars(title, all_metrics):
    fig = go.Figure()
    for agent_name, metrics in all_metrics.items():
        mean_vs = {k: np.mean(values) for k, values in metrics.items()}
        std_vs = {k: np.std(values) for k, values in metrics.items()}
        x = list(mean_vs.keys())
        y = list(mean_vs.values())
        e = list(std_vs.values())
        fig.add_trace(
            go.Bar(name=agent_name, x=x, y=y, error_y=dict(type="data", array=e),)
        )

    fig.update_layout(
        title=title, xaxis_title="Metric", yaxis_title="Value", barmode="group"
    )
    fig.show()


def main():
    n_episodes = 10
    agent_names = ["pure_pursuit", "stanley", "lqr"]
    render_mode = None

    logging.basicConfig(level=logging.INFO)

    task = TaskControl(render_mode=render_mode)
    all_metrics = {}
    for agent_name in agent_names:
        task.reset()
        agent = F110Agent(name=agent_name)

        runner = Runner(task=task, agent=agent)
        all_metrics[agent_name] = runner.run(n_episodes=n_episodes)

    plot_metrics_bars(title="Benchmark Control", all_metrics=all_metrics)


if __name__ == "__main__":
    main()
