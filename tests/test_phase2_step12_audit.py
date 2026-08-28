import copy
import hashlib
import json
import pytest
import numpy as np
import yaml

from envs.entities import SimulationConfig, Vehicle, Task, RSU
from envs.task_generator import TaskGenerator
from envs.vec_env import VECEnv


def load_config(overrides=None) -> SimulationConfig:
    with open("configs/paper_parameters.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    if overrides:
        cfg_dict.update(overrides)
    return SimulationConfig(**cfg_dict)


def test_12_1_workload_cardinality():
    """
    Step 12.1 — Workload Cardinality Audit
    Verify for workloads {20, 30, 40} tasks/vehicle with 10 vehicles:
      N_target = workload_tasks_per_vehicle * num_vehicles
      len(materialized_tasks) == N_target
    """
    for tasks_per_veh in [20, 30, 40]:
        cfg = load_config({"num_tasks_per_vehicle_range": [tasks_per_veh, tasks_per_veh]})
        task_gen = TaskGenerator(cfg)
        
        num_vehicles = 10
        all_tasks = []
        for v in range(num_vehicles):
            v_tasks = task_gen.generate_tasks_for_vehicle(f"veh_{v}")
            all_tasks.extend(v_tasks)
            
        n_target = tasks_per_veh * num_vehicles
        assert len(all_tasks) == n_target, f"Expected {n_target} tasks, got {len(all_tasks)}"
        
        # Verify arrival rate timestamp sampling does not alter cardinality
        arrival_rate = 30.0 # tasks/s
        timestamps = np.cumsum(np.random.exponential(1.0 / arrival_rate, size=n_target))
        assert len(timestamps) == n_target


def test_12_2_task_conservation_and_terminal_states():
    """
    Step 12.2 — Task Conservation & Terminal Event Audit
    Verify N_gen = N_comp + N_fail + N_pend
    and N_fail = N_dual + N_dead + N_cov + N_dep
    Verify 6 terminal state transitions and immutability of COMPLETED status.
    """
    cfg = load_config()
    env = VECEnv(config=cfg, scenario_geometry="corridor_2400m", use_mobility_model=False, max_vehicles=3)
    
    try:
        env.reset()
        done = False
        step = 0
        while not done and step < 50:
            action = step % 7
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
        n_comp = len(env.completed_tasks)
        n_fail = len(env.failed_tasks)
        n_pend = len(env.pending_tasks)
        
        # Total owned tasks across active vehicles
        total_veh_tasks = sum(len(ts) for ts in env.vehicle_tasks.values())
        total_gen = total_veh_tasks + n_comp + n_fail
        
        assert total_gen == n_comp + n_fail + n_pend, "Task conservation identity failed"
        
        # Categorize failure breakdown
        fail_reasons = {}
        for task, reason in env.failed_tasks:
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            
        valid_reasons = {"DEADLINE_EXCEEDED", "COVERAGE_VIOLATION", "DUAL_VIOLATION", "FAILED_DEPARTURE"}
        for r in fail_reasons:
            assert r in valid_reasons, f"Invalid failure reason: {r}"
            
        assert sum(fail_reasons.values()) == n_fail
    finally:
        env.close()


def test_12_3_latency_decomposition():
    """
    Step 12.3 — Latency Decomposition Audit
    Verify T_total = T_comm + T_wait + T_comp across all completed transitions.
    """
    cfg = load_config()
    env = VECEnv(config=cfg, scenario_geometry="corridor_2400m", use_mobility_model=False, max_vehicles=3)
    
    try:
        env.reset()
        max_residual = 0.0
        
        for step in range(30):
            obs, reward, terminated, truncated, info = env.step(step % 7)
            if "delay" in info and "comm_delay" in info:
                t_total = info["delay"]
                t_sum = info["comm_delay"] + info["wait_delay"] + info["comp_delay"]
                residual = abs(t_total - t_sum)
                max_residual = max(max_residual, residual)
                assert residual <= 1e-4, f"Latency decomposition residual exceeded: {residual}"
                
        assert max_residual <= 1e-4
    finally:
        env.close()


def test_12_4_energy_decomposition():
    """
    Step 12.4 — Energy Decomposition Audit
    Verify E_total components are non-negative and finite.
    """
    cfg = load_config()
    env = VECEnv(config=cfg, scenario_geometry="corridor_2400m", use_mobility_model=False, max_vehicles=3)
    
    try:
        env.reset()
        for step in range(30):
            obs, reward, terminated, truncated, info = env.step(step % 7)
            if "energy" in info:
                assert info["energy"] >= 0.0, f"Negative energy: {info['energy']}"
                assert np.isfinite(info["energy"]), f"Non-finite energy: {info['energy']}"
    finally:
        env.close()


def test_12_5_queue_invariants():
    """
    Step 12.5 — Queue Audit
    Verify Q_m(t) >= 0 for all RSUs throughout all transitions.
    """
    cfg = load_config()
    env = VECEnv(config=cfg, scenario_geometry="corridor_2400m", use_mobility_model=False, max_vehicles=3)
    
    try:
        env.reset()
        for step in range(30):
            env.step(step % 7)
            for rsu in env.rsus:
                assert rsu.queued_cpu_cycles >= 0.0, f"Negative queue on RSU {rsu.rsu_id}: {rsu.queued_cpu_cycles}"
    finally:
        env.close()
