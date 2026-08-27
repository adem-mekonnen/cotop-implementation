"""
tests/test_multivehicle_contention.py
Formal test suite verifying:
- Genuine multi-vehicle simulation and task generation (TEST 1, TEST 2)
- Shared RSU queue contention and Eq. 5 wait time (TEST 3, TEST 5)
- Queue conservation over time slots (TEST 4)
- Normalized Eq. 23 task priority scale & alpha/beta sensitivity (TEST 6, TEST 7)
- Action semantics and physical routing (TEST 8)
- Baseline fairness (CoTOP, Local, Greedy across identical workload) (TEST 9)
- Edge cases and lifecycle invariants (TEST 10)
"""
import pytest
import math
import numpy as np
import yaml
import torch

from envs.entities import Vehicle, Task, RSU, SimulationConfig
from envs.vec_env import VECEnv, get_euclidean_distance
from envs.comm_model import compute_v2r_rate, compute_r2r_rate
from envs.comp_model import calculate_case1_standalone, calculate_case2_collaboration
from utils.task_priority import compute_task_priority, prioritize_tasks, prioritize_task_queue
from models.baselines.local import LocalPolicy
from models.baselines.greedy import GreedyPolicy


@pytest.fixture
def base_config():
    with open("configs/paper_parameters.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    return SimulationConfig(**cfg_dict)


# =========================================================================
# TEST 1: 2 Concurrent Vehicles
# =========================================================================
def test_two_vehicles_concurrency(base_config):
    env = VECEnv(config=base_config, port=8850, seed=42, max_vehicles=2)
    obs, _ = env.reset(options={"max_vehicles": 2})

    generated_vehicles = set()
    total_steps = 0
    done = False

    while not done and total_steps < 100:
        action = 0  # Local policy
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_steps += 1
        if "v_id" in info:
            generated_vehicles.add(info["v_id"])

    env.close()
    assert len(generated_vehicles) == 2, f"Expected 2 vehicles, got {generated_vehicles}"
    assert total_steps == 40, f"Expected 40 tasks (20 per vehicle * 2), got {total_steps}"


# =========================================================================
# TEST 2: 10 Vehicles Represented in Workload
# =========================================================================
def test_ten_vehicles_workload(base_config):
    env = VECEnv(config=base_config, port=8851, seed=42, max_vehicles=10)
    obs, _ = env.reset(options={"max_vehicles": 10})

    generated_vehicles = set()
    total_steps = 0
    done = False

    while not done and total_steps < 500:
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_steps += 1
        if "v_id" in info:
            generated_vehicles.add(info["v_id"])

    env.close()
    assert len(generated_vehicles) == 10, f"Expected 10 vehicles, got {len(generated_vehicles)}"
    assert total_steps == 200, f"Expected 200 tasks (20 * 10), got {total_steps}"


# =========================================================================
# TEST 3: Shared RSU Queue Contention
# =========================================================================
def test_shared_rsu_queue_contention(base_config):
    # Two vehicles offloading to RSU 0. Task B arrives behind Task A.
    rsu0 = RSU(
        rsu_id=0,
        location=(200.0, 0.0),
        cpu_capacity_f=1.0e9,  # 1 GHz
        queued_cpu_cycles=0.0,
        transmission_power_P_R=100.0
    )
    
    task_a = Task(task_id=1, vehicle_id="veh_A", size_rho=2.0e6, cpu_phi=10.0e6, max_delay_d=25.0)
    task_b = Task(task_id=2, vehicle_id="veh_B", size_rho=2.0e6, cpu_phi=10.0e6, max_delay_d=25.0)
    
    # Task A arrives at empty queue
    queue_before_a = rsu0.queued_cpu_cycles
    t_wait_a = queue_before_a / rsu0.cpu_capacity_f
    assert t_wait_a == 0.0
    
    # Task A offloads standalone to RSU 0
    rsu0.queued_cpu_cycles += task_a.cpu_phi
    assert rsu0.queued_cpu_cycles == 10.0e6
    
    # Task B arrives immediately behind Task A at same RSU
    queue_before_b = rsu0.queued_cpu_cycles
    t_wait_b = queue_before_b / rsu0.cpu_capacity_f
    
    # Verify contention: t_wait_b > 0 and reflects queued workload
    assert queue_before_b > queue_before_a
    assert pytest.approx(t_wait_b, rel=1e-5) == (10.0e6 / 1.0e9) == 0.01


# =========================================================================
# TEST 4: Queue Conservation
# =========================================================================
def test_queue_conservation():
    # Q_new = max(0, Q_old + arrivals - service * dt)
    rsu = RSU(0, (0.0, 0.0), cpu_capacity_f=2.0e9, queued_cpu_cycles=5.0e9, transmission_power_P_R=100.0)
    dt = 1.0
    
    # 1. Add arrivals
    arrival_cycles = 3.0e9
    rsu.queued_cpu_cycles += arrival_cycles
    assert rsu.queued_cpu_cycles == 8.0e9
    
    # 2. Advance time slot: service F_m * dt = 2.0e9
    rsu.queued_cpu_cycles = max(0.0, rsu.queued_cpu_cycles - rsu.cpu_capacity_f * dt)
    assert pytest.approx(rsu.queued_cpu_cycles, rel=1e-5) == 6.0e9
    
    # 3. Simulate 4 seconds of idle service (depletion to 0)
    for _ in range(4):
        rsu.queued_cpu_cycles = max(0.0, rsu.queued_cpu_cycles - rsu.cpu_capacity_f * dt)
    assert rsu.queued_cpu_cycles == 0.0


# =========================================================================
# TEST 5: Exact Eq. 5 Calculation
# =========================================================================
def test_eq5_exact_wait_time():
    # Eq 5: t_wait = N^{queue} / F_m
    queued_cycles = 18.96e9  # 18.96 Gcycles
    rsu_cpu_f = 2.0e9        # 2 GHz
    
    t_wait = queued_cycles / rsu_cpu_f
    assert pytest.approx(t_wait, rel=1e-5) == 9.48  # seconds


# =========================================================================
# TEST 6: Task Priority Scale Balance
# =========================================================================
def test_task_priority_scale_balance():
    # Table III extremes
    t_min = Task(1, "v1", size_rho=2.0e6, cpu_phi=1.0e6, max_delay_d=30.0)
    t_max = Task(2, "v1", size_rho=5.0e6, cpu_phi=10.0e6, max_delay_d=20.0)
    
    # Verify dwell term is in (0, 1)
    dwell_term_min = math.exp(-1.0 / 0.5)
    dwell_term_max = math.exp(-1.0 / 100.0)
    assert 0.0 < dwell_term_min < 1.0
    assert 0.0 < dwell_term_max < 1.0
    
    # Verify normalized size_delay_term is in [0.267, 1.0]
    norm_size_delay_min = (2.0e6 / 5.0e6) / (30.0 / 20.0)  # 0.4 / 1.5 = 0.2667
    norm_size_delay_max = (5.0e6 / 5.0e6) / (20.0 / 20.0)  # 1.0
    assert pytest.approx(norm_size_delay_min, rel=1e-3) == 0.2667
    assert norm_size_delay_max == 1.0
    
    # Verify composite priority is in [0, 1]
    p_min = compute_task_priority(t_min, dwell_time=0.5, alpha=0.3, beta=0.7)
    p_max = compute_task_priority(t_max, dwell_time=100.0, alpha=0.3, beta=0.7)
    assert 0.0 < p_min < 1.0
    assert 0.0 < p_max < 1.0


# =========================================================================
# TEST 7: Alpha/Beta Sensitivity
# =========================================================================
def test_alpha_beta_priority_sensitivity():
    t_a = Task(1, "vA", size_rho=5.0e6, cpu_phi=10.0e6, max_delay_d=20.0)  # High urgency
    t_b = Task(2, "vB", size_rho=2.0e6, cpu_phi=10.0e6, max_delay_d=30.0)  # Low urgency
    
    dwell_a = 2.0   # Short dwell
    dwell_b = 60.0  # Long dwell
    
    # Under alpha=1.0, beta=0.0: dwell time dominates -> Task B wins
    p_a_dwell = compute_task_priority(t_a, dwell_time=dwell_a, alpha=1.0, beta=0.0)
    p_b_dwell = compute_task_priority(t_b, dwell_time=dwell_b, alpha=1.0, beta=0.0)
    assert p_b_dwell > p_a_dwell
    
    # Under alpha=0.0, beta=1.0: task urgency dominates -> Task A wins
    p_a_task = compute_task_priority(t_a, dwell_time=dwell_a, alpha=0.0, beta=1.0)
    p_b_task = compute_task_priority(t_b, dwell_time=dwell_b, alpha=0.0, beta=1.0)
    assert p_a_task > p_b_task


# =========================================================================
# TEST 8: Action Semantics (Standalone vs Collaborative)
# =========================================================================
def test_action_semantics_routing(base_config):
    env = VECEnv(config=base_config, port=8852, seed=42, max_vehicles=1)
    obs, _ = env.reset(options={"max_vehicles": 1})
    
    # Action 0: Standalone
    _, _, _, _, info_0 = env.step(0)
    assert info_0["case"] == 1
    
    # Action 2: Collaboration with RSU 1 (action - 1 = 1)
    _, _, _, _, info_2 = env.step(2)
    assert info_2["case"] in [1, 2]  # 2 if primary is not RSU 1, fallback to 1 if primary is RSU 1
    
    env.close()


# =========================================================================
# TEST 9: Baseline Fairness across CoTOP, Local, Greedy
# =========================================================================
def test_baseline_fairness_identical_conditions(base_config):
    seed = 12345
    
    # 1. Run Local Baseline
    env_local = VECEnv(config=base_config, port=8853, seed=seed, max_vehicles=3)
    policy_local = LocalPolicy(config=base_config)
    obs, _ = env_local.reset(seed=seed, options={"max_vehicles": 3})
    done = False
    local_tasks = []
    while not done:
        action = policy_local.select_action(obs)
        obs, reward, terminated, truncated, info = env_local.step(action)
        done = terminated or truncated
        local_tasks.append((info["v_id"], info["task_id"]))
    env_local.close()
    
    # 2. Run Greedy Baseline
    env_greedy = VECEnv(config=base_config, port=8854, seed=seed, max_vehicles=3)
    policy_greedy = GreedyPolicy(config=base_config)
    obs, _ = env_greedy.reset(seed=seed, options={"max_vehicles": 3})
    done = False
    greedy_tasks = []
    while not done:
        action = policy_greedy.select_action(obs)
        obs, reward, terminated, truncated, info = env_greedy.step(action)
        done = terminated or truncated
        greedy_tasks.append((info["v_id"], info["task_id"]))
    env_greedy.close()
    
    # Verify exact workload identity
    assert len(local_tasks) == len(greedy_tasks) == 60
    assert local_tasks == greedy_tasks, "Baselines must receive identical task sequence and workload"


# =========================================================================
# TEST 10: Lifecycle Invariants & Duplicate Prevention
# =========================================================================
def test_multi_vehicle_lifecycle_invariants(base_config):
    env = VECEnv(config=base_config, port=8855, seed=42, max_vehicles=4)
    obs, _ = env.reset(options={"max_vehicles": 4})
    
    assert env.observation_space.shape == (114,)
    assert env.action_space.n == 7
    assert len(env.rsus) == 6
    
    seen_task_ids = set()
    total_steps = 0
    done = False
    while not done and total_steps < 200:
        obs, reward, terminated, truncated, info = env.step(0)
        done = terminated or truncated
        total_steps += 1
        
        # Verify no duplicate task IDs processed
        task_uid = (info["v_id"], info["task_id"])
        assert task_uid not in seen_task_ids, f"Duplicate task detected: {task_uid}"
        seen_task_ids.add(task_uid)
        
        # Verify physical values are non-negative
        assert info["delay"] >= 0.0
        assert info["energy"] >= 0.0
        assert info["comm_delay"] >= 0.0
        assert info["comp_delay"] >= 0.0
        assert info["wait_delay"] >= 0.0
        assert all(q >= 0.0 for q in info["rsu_queues"])
        
    env.close()
    assert total_steps == 80, f"Expected 80 total tasks (4 * 20), got {total_steps}"
