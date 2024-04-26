from f1tenth_benchmark.metrics.basics import StepCounter
from f1tenth_benchmark.core.trace_manager import TraceManager
import pytest

def test_abstract_implementation():
    mc = StepCounter()

def test_step_counter_function():
    mc = StepCounter()
    tm = TraceManager()
    mc.register_trace_manager(tm)
    for i in range(10):
        mc.evaluate(None)
        assert mc.get_final_score() == i + 1
    
    mc.reset()
    assert mc.get_final_score() == 0
        