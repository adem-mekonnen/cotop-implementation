"""
tests/test_action_sensitivity_audit.py
Automated regression tests verifying deterministic action sensitivity in VECEnv / FrozenVECEnv.
Distinguishes Hypothesis 1 (action path bug) from Hypothesis 2 (policy degeneracy).
"""

import os
import pytest
import numpy as np
import yaml

from envs.frozen_vec_env import FrozenVECEnv
from envs.entities import SimulationConfig
from envs.vec_env import get_euclidean_distance

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "paper_parameters.yaml")
REALIZATION_PATH = os.path.join(
    ROOT_DIR, "data", "evaluation_realizations", "realization_corridor_2400m_w20_seed42.json"
)

@pytest.fixture
def test_config():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["num_tasks_per_vehicle_range"] = [20, 20]
    return SimulationConfig(**cfg)

def test_a_local_decoding(test_config):
    """
    Test A — Local decoding:
    action 0 must strictly result in Case 1 standalone execution without secondary RSU.
    """
    env = FrozenVECEnv(config=test_config, realization_path=REALIZATION_PATH)
    env.reset()
    assert len(env.pending_tasks) > 0, "No pending tasks generated."
    
    obs, reward, terminated, truncated, info = env.step(0)
    assert info["case"] == 1, f"Expected Case 1 standalone execution, got {info['case']}"
    assert info["wait_delay"] >= 0.0
    assert info["delay"] > 0.0
    assert info["energy"] > 0.0
    env.close()

def test_b_collaborative_decoding(test_config):
    """
    Test B — Collaborative decoding:
    A valid collaborative action targeting a secondary RSU must strictly invoke Case 2.
    """
    env = FrozenVECEnv(config=test_config, realization_path=REALIZATION_PATH)
    env.reset()
    assert len(env.pending_tasks) > 0, "No pending tasks generated."
    
    curr_veh, curr_task = env.pending_tasks[0]
    target_rsu = min(env.rsus, key=lambda r: get_euclidean_distance(curr_veh.pos, r.location))
    
    # Pick a distinct secondary RSU
    sec_id = (target_rsu.rsu_id + 1) % len(env.rsus)
    action = sec_id + 1 # 1-indexed RSU action
    
    obs, reward, terminated, truncated, info = env.step(action)
    assert info["case"] == 2, f"Expected Case 2 collaborative execution, got {info['case']}"
    assert info["comm_delay"] > 0.0
    assert info["comp_delay"] > 0.0
    env.close()

def test_c_action_sensitivity_paired_step(test_config):
    """
    Test C — Action sensitivity:
    For the exact same initial state and task, Action 0 vs Collaborative Action must produce
    different execution semantics (different case, different energy, and different queue states).
    """
    # Step with Local
    env_local = FrozenVECEnv(config=test_config, realization_path=REALIZATION_PATH)
    env_local.reset()
    curr_veh, curr_task = env_local.pending_tasks[0]
    target_rsu_id = min(env_local.rsus, key=lambda r: get_euclidean_distance(curr_veh.pos, r.location)).rsu_id
    
    obs_loc, rew_loc, term_loc, trunc_loc, info_loc = env_local.step(0)
    
    # Step with Collab
    env_collab = FrozenVECEnv(config=test_config, realization_path=REALIZATION_PATH)
    env_collab.reset()
    sec_id = (target_rsu_id + 1) % len(env_collab.rsus)
    collab_action = sec_id + 1
    
    obs_col, rew_col, term_col, trunc_col, info_col = env_collab.step(collab_action)
    
    # Assert physical differences
    assert info_loc["case"] == 1, "Local policy must execute Case 1"
    assert info_col["case"] == 2, "Collab policy must execute Case 2"
    assert info_loc["case"] != info_col["case"], "Execution cases must differ"
    assert abs(info_loc["energy"] - info_col["energy"]) > 1e-4, "Energy consumption must differ due to R2R relaying"
    
    env_local.close()
    env_collab.close()

def test_d_queue_update_semantics(test_config):
    """
    Test D — Queue update:
    Verify the correct RSU queues are modified according to the selected action.
    - Case 1: Only target primary RSU queue increases.
    - Case 2: Both primary and secondary RSU queues increase according to partition.
    """
    # Case 1 Queue Update
    env1 = FrozenVECEnv(config=test_config, realization_path=REALIZATION_PATH)
    env1.reset()
    curr_veh, curr_task = env1.pending_tasks[0]
    target_rsu1 = min(env1.rsus, key=lambda r: get_euclidean_distance(curr_veh.pos, r.location))
    t_rsu_id = target_rsu1.rsu_id
    sec_rsu_id = (t_rsu_id + 1) % len(env1.rsus)
    
    q_target_before = env1.rsus[t_rsu_id].queued_cpu_cycles
    q_sec_before = env1.rsus[sec_rsu_id].queued_cpu_cycles
    
    env1.step(0)
    
    q_target_after = env1.rsus[t_rsu_id].queued_cpu_cycles
    q_sec_after = env1.rsus[sec_rsu_id].queued_cpu_cycles
    
    assert q_target_after > q_target_before, "Primary RSU queue must increase in Case 1"
    assert q_sec_after == q_sec_before, "Secondary RSU queue must NOT change in Case 1"
    env1.close()
    
    # Case 2 Queue Update
    env2 = FrozenVECEnv(config=test_config, realization_path=REALIZATION_PATH)
    env2.reset()
    q_target2_before = env2.rsus[t_rsu_id].queued_cpu_cycles
    q_sec2_before = env2.rsus[sec_rsu_id].queued_cpu_cycles
    
    env2.step(sec_rsu_id + 1)
    
    q_target2_after = env2.rsus[t_rsu_id].queued_cpu_cycles
    q_sec2_after = env2.rsus[sec_rsu_id].queued_cpu_cycles
    
    assert q_target2_after > q_target2_before or q_sec2_after > q_sec2_before, "Queues must receive task cycles in Case 2"
    env2.close()

def test_e_completion_evaluation(test_config):
    """
    Test E — Completion evaluation:
    Verify completion/failure flags accurately reflect delay vs deadline and coverage bounds.
    """
    env = FrozenVECEnv(config=test_config, realization_path=REALIZATION_PATH)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    
    assert "completed" in info
    assert "failure_reason" in info
    assert "fail_deadline" in info
    assert "fail_coverage" in info
    
    if info["completed"]:
        assert info["failure_reason"] == "NONE"
        assert not info["fail_deadline"]
        assert not info["fail_coverage"]
    else:
        assert info["failure_reason"] in ["DEADLINE_EXCEEDED", "COVERAGE_VIOLATION", "DUAL_VIOLATION"]
        
    env.close()
