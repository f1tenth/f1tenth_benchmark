from f1tenth_benchmark.agent.agent import Agent


def make_f110_agent(agent_name: str) -> Agent:
    if agent_name == "pure_pursuit":
        pass
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")