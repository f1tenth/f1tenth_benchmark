from f1tenth_benchmark.agent.f110_agent import F110Agent
from f1tenth_benchmark.runner import Runner
from f1tenth_benchmark.task.task_control import TaskControl


def main():
    n_episodes = 10
    agent_name = "pure_pursuit"

    task = TaskControl()
    agent = F110Agent(name=agent_name)

    runner = Runner(task=task, agent=agent, log_dir="logs")
    runner.run(n_episodes=n_episodes)





if __name__=="__main__":
    main()