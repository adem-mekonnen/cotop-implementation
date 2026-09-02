"""
tests/test_completion_failure_semantics.py
Automated regression tests verifying completion, failure, coverage, deadline, and reward semantics.
"""

import os
import pytest
import numpy as np
import yaml

from envs.frozen_vec_env import FrozenVECEnv
from envs.entities import SimulationConfig, Task, Vehicle, RSU
from envs.vec_env import VECEnv, get_euclidean_distance

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "paper_parameters.yaml")
REALIZATION_PATH = os.path.join(
    ROOT_DIR, "data", "evaluation_realizations", "realization_corridor_2400m_w20_seed42.json"
)

@pytest.fixture
def base_config():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["num_tasks_per_vehicle_range"] = [20, 20]
    return SimulationConfig(**cfg)

def test_a_task_completion_nominal(base_config):
    """Test A: Task completes when finished before deadline and within coverage."""
    env = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    
    assert info["completed"] is True
    assert info["fail_deadline"] is False
    assert info["fail_coverage"] is False
    assert info["failure_reason"] == "NONE"
    assert reward < 0.0 # Nominal negative delay/energy cost, not penalty
    assert reward > -base_config.penalty_z
    env.close()

def test_b_task_failure_deadline_exceeded(base_config):
    """Test B: Task fails when execution delay exceeds deadline."""
    env = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    env.reset()
    
    # Force tiny deadline on top task
    curr_veh, curr_task = env.pending_tasks[0]
    curr_task.max_delay_d = 0.001 # 1 ms deadline
    
    obs, reward, terminated, truncated, info = env.step(0)
    
    assert info["completed"] is False
    assert info["fail_deadline"] is True
    assert info["failure_reason"] in ["DEADLINE_EXCEEDED", "DUAL_VIOLATION"]
    assert reward == -base_config.penalty_z
    env.close()

def test_c_task_failure_coverage_violation(base_config):
    """Test C: Task fails when vehicle exits coverage radius during execution."""
    env = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    env.reset()
    
    # Place vehicle right at the edge of RSU coverage moving at high speed
    curr_veh, curr_task = env.pending_tasks[0]
    curr_veh.pos = (399.0, 0.0) # Near edge of RSU 0 (range 400.0)
    curr_veh.speed = 100.0 # Very high speed to exit coverage immediately
    
    obs, reward, terminated, truncated, info = env.step(0)
    
    assert info["completed"] is False
    assert info["fail_coverage"] is True
    assert info["failure_reason"] in ["COVERAGE_VIOLATION", "DUAL_VIOLATION"]
    assert reward == -base_config.penalty_z
    env.close()

def test_d_distinguishable_failure_reasons(base_config):
    """Test D: Coverage failure and deadline failure are explicitly distinguishable in info."""
    env = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    env.reset()
    
    # 1. Deadline failure only
    curr_veh, curr_task = env.pending_tasks[0]
    curr_veh.pos = (200.0, 0.0) # Right at RSU 0 center
    curr_veh.speed = 0.0001 # Zero speed -> will not leave coverage
    curr_task.max_delay_d = 0.0001 # Extremely small deadline
    
    _, _, _, _, info_deadline = env.step(0)
    assert info_deadline["fail_deadline"] is True
    assert info_deadline["fail_coverage"] is False
    assert info_deadline["failure_reason"] == "DEADLINE_EXCEEDED"
    
    env.close()

def test_e_case1_uses_standalone_delay(base_config):
    """Test E: Case 1 completion uses Case 1 execution delay."""
    env = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    env.reset()
    _, _, _, _, info = env.step(0)
    
    assert info["case"] == 1
    assert abs(info["delay"] - (info["comm_delay"] + info["comp_delay"] + info["wait_delay"])) < 1e-5
    env.close()

def test_f_case2_uses_collaborative_delay(base_config):
    """Test F: Case 2 completion uses Case 2 parallel execution delay."""
    env = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    env.reset()
    
    curr_veh, curr_task = env.pending_tasks[0]
    target_rsu = min(env.rsus, key=lambda r: get_euclidean_distance(curr_veh.pos, r.location))
    sec_id = (target_rsu.rsu_id + 1) % len(env.rsus)
    
    _, _, _, _, info = env.step(sec_id + 1)
    
    assert info["case"] == 2
    assert info["delay"] > 0.0
    assert info["comm_delay"] > 0.0
    env.close()

def test_g_different_completion_outcomes_by_case(base_config):
    """
    Test G: Same task under Case 1 and Case 2 can produce different completion outcomes
    when delay differences cause a predicate boundary crossing.
    """
    # Create an artificial scenario where Case 1 deadline fails, but Case 2 completes before deadline
    env1 = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    env1.reset()
    curr_veh, curr_task = env1.pending_tasks[0]
    curr_veh.pos = (200.0, 0.0)
    curr_veh.speed = 0.0
    
    # When deadline is 1ms (less than ~50ms transmission delay), it fails deadline:
    curr_task.max_delay_d = 0.001 # 1ms deadline
    
    _, _, _, _, info_c1 = env1.step(0)
    assert info_c1["case"] == 1
    assert info_c1["fail_deadline"] is True # Delay > 1ms -> Fails
    assert info_c1["completed"] is False
    
    # If we set deadline to be ample (10.0s), it completes
    env2 = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    env2.reset()
    curr_veh2, curr_task2 = env2.pending_tasks[0]
    curr_veh2.pos = (200.0, 0.0)
    curr_veh2.speed = 0.0
    curr_task2.max_delay_d = 10.0 # 10.0s deadline
    
    _, _, _, _, info_c2 = env2.step(2)
    assert info_c2["case"] == 2
    assert info_c2["fail_deadline"] is False
    assert info_c2["completed"] is True
    
    env1.close()
    env2.close()

def test_h_explicit_boundary_exit_predicate(base_config):
    """Test H: Explicit test of the corridor boundary exit at x = 2400m."""
    env = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    env.reset()
    
    # Place vehicle at x=2400m with 35 m/s speed
    curr_veh, curr_task = env.pending_tasks[0]
    curr_veh.pos = (2400.0, 0.0)
    curr_veh.speed = 35.0
    curr_task.max_delay_d = 30.0 # Ample deadline
    
    _, _, _, _, info = env.step(0)
    
    assert info["fail_coverage"] is True
    assert info["fail_deadline"] is False
    assert info["failure_reason"] == "COVERAGE_VIOLATION"
    assert info["completed"] is False
    env.close()
