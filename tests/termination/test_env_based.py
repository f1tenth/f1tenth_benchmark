from f1tenth_benchmark.termination.env_based import TimeStepTermination, EnvTermination



def test_abstract_implementation():
    tc = TimeStepTermination(agent_id=0, max_time_steps=100)
    tc = EnvTermination()

