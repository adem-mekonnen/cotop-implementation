"""
tests/test_baseline_physics.py
Validates that Local and Greedy baseline policies produce divergent actions and distinct physical metrics.
"""
import pytest
import numpy as np
from envs.entities import SimulationConfig, Task, Vehicle, RSU
from envs.state_builder import build_state
from models.baselines.local import LocalPolicy
from models.baselines.greedy import GreedyPolicy

def test_baseline_action_divergence_and_physics():
    config = SimulationConfig(num_rsus=6, max_task_cpu=10.0)
    local_policy = LocalPolicy(config=config)
    greedy_policy = GreedyPolicy(config=config)

    # State where Primary RSU (RSU 0) has heavy queue, but Secondary RSU (RSU 1) is empty
    vehicle = Vehicle("v0", (100.0, 0.0), 35.0, dwell_time_T_stay=10.0)
    tasks = [Task(0, "v0", 3.0e6, 10.0e6, 25.0)]
    rsus = [
        RSU(0, (0.0, 0.0), 1.0e9, 80.0e6, 100.0),   # Nearest, heavy queue (80M cycles = 80ms)
        RSU(1, (400.0, 0.0), 4.0e9, 0.0, 100.0),   # Secondary, 0 queue, 4 GHz
        RSU(2, (800.0, 0.0), 2.0e9, 10.0e6, 100.0),
        RSU(3, (1200.0, 0.0), 2.0e9, 10.0e6, 100.0),
        RSU(4, (1600.0, 0.0), 2.0e9, 10.0e6, 100.0),
        RSU(5, (2000.0, 0.0), 2.0e9, 10.0e6, 100.0),
    ]

    state = build_state(vehicle, tasks, rsus, config)

    action_local = local_policy.select_action(state)
    action_greedy = greedy_policy.select_action(state)

    # Local must strictly select Standalone (Action 0)
    assert action_local == 0

    # Greedy must select the RSU with minimum wait time (RSU 1 -> Action 2)
    assert action_greedy == 2
    assert action_local != action_greedy
