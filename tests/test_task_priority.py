import pytest
import math
from envs.entities import Task
from utils.task_priority import compute_task_priority, prioritize_tasks, prioritize_task_queue


def test_task_priority_ordering():
    # Stricter deadline -> higher priority
    # Larger data volume -> higher priority
    t1 = Task(task_id=1, vehicle_id="v1", size_rho=5.0e6, cpu_phi=10.0e6, max_delay_d=20.0)  # High size, strict deadline
    t2 = Task(task_id=2, vehicle_id="v1", size_rho=2.0e6, cpu_phi=10.0e6, max_delay_d=30.0)  # Low size, relaxed deadline

    dwell_time = 15.0
    p1 = compute_task_priority(t1, dwell_time, alpha=0.3, beta=0.7)
    p2 = compute_task_priority(t2, dwell_time, alpha=0.3, beta=0.7)
    assert p1 > p2

    sorted_tasks = prioritize_tasks([t2, t1], dwell_time, alpha=0.3, beta=0.7)
    assert sorted_tasks[0].task_id == 1
    assert sorted_tasks[1].task_id == 2


def test_task_priority_normalized_scale_bounds():
    # Under Table III parameters:
    # size_rho in [2.0e6, 5.0e6], max_delay_d in [20.0, 30.0], dwell_time in (0, 100]
    # Both dwell_term and size_delay_term must be dimensionless and bounded in [0, 1]
    t_min = Task(task_id=1, vehicle_id="v1", size_rho=2.0e6, cpu_phi=10.0e6, max_delay_d=30.0)
    t_max = Task(task_id=2, vehicle_id="v1", size_rho=5.0e6, cpu_phi=10.0e6, max_delay_d=20.0)

    p_low = compute_task_priority(t_min, dwell_time=0.5, alpha=0.3, beta=0.7)
    p_high = compute_task_priority(t_max, dwell_time=100.0, alpha=0.3, beta=0.7)

    assert 0.0 <= p_low <= 1.0
    assert 0.0 <= p_high <= 1.0
    assert p_high > p_low


def test_task_priority_alpha_beta_sensitivity():
    # When alpha=1, beta=0: priority is purely determined by dwell time
    # When alpha=0, beta=1: priority is purely determined by task urgency (size/deadline)
    t_urgent = Task(task_id=1, vehicle_id="v1", size_rho=5.0e6, cpu_phi=10.0e6, max_delay_d=20.0)
    t_relaxed = Task(task_id=2, vehicle_id="v2", size_rho=2.0e6, cpu_phi=10.0e6, max_delay_d=30.0)

    # Vehicle 1 has short dwell time (5s), Vehicle 2 has long dwell time (50s)
    # With alpha=1, beta=0: Vehicle 2's task should have HIGHER priority due to dwell time
    p_v1_dwell = compute_task_priority(t_urgent, dwell_time=5.0, alpha=1.0, beta=0.0)
    p_v2_dwell = compute_task_priority(t_relaxed, dwell_time=50.0, alpha=1.0, beta=0.0)
    assert p_v2_dwell > p_v1_dwell

    # With alpha=0, beta=1: Vehicle 1's task should have HIGHER priority due to size/deadline
    p_v1_task = compute_task_priority(t_urgent, dwell_time=5.0, alpha=0.0, beta=1.0)
    p_v2_task = compute_task_priority(t_relaxed, dwell_time=50.0, alpha=0.0, beta=1.0)
    assert p_v1_task > p_v2_task


def test_task_priority_monotonicity():
    dwell = 10.0
    # 1. Monotonicity in size: larger size -> higher priority
    t_small = Task(task_id=1, vehicle_id="v", size_rho=2.0e6, cpu_phi=10e6, max_delay_d=25.0)
    t_large = Task(task_id=2, vehicle_id="v", size_rho=4.0e6, cpu_phi=10e6, max_delay_d=25.0)
    assert compute_task_priority(t_large, dwell) > compute_task_priority(t_small, dwell)

    # 2. Monotonicity in deadline: stricter deadline (smaller d) -> higher priority
    t_urgent = Task(task_id=3, vehicle_id="v", size_rho=3.0e6, cpu_phi=10e6, max_delay_d=20.0)
    t_relaxed = Task(task_id=4, vehicle_id="v", size_rho=3.0e6, cpu_phi=10e6, max_delay_d=30.0)
    assert compute_task_priority(t_urgent, dwell) > compute_task_priority(t_relaxed, dwell)

    # 3. Monotonicity in dwell time: longer dwell time -> higher priority
    t = Task(task_id=5, vehicle_id="v", size_rho=3.0e6, cpu_phi=10e6, max_delay_d=25.0)
    assert compute_task_priority(t, dwell_time=30.0) > compute_task_priority(t, dwell_time=5.0)


def test_task_priority_edge_cases():
    # Non-positive dwell time should yield 0.0 dwell term
    t = Task(task_id=1, vehicle_id="v", size_rho=3.0e6, cpu_phi=10e6, max_delay_d=25.0)
    p_zero_dwell = compute_task_priority(t, dwell_time=0.0, alpha=0.3, beta=0.7)
    p_neg_dwell = compute_task_priority(t, dwell_time=-5.0, alpha=0.3, beta=0.7)
    expected_task_term = 0.7 * ((3.0e6 / 5.0e6) / (25.0 / 20.0))
    assert pytest.approx(p_zero_dwell, rel=1e-5) == expected_task_term
    assert pytest.approx(p_neg_dwell, rel=1e-5) == expected_task_term

    # Non-positive deadline should yield 0.0 size_delay term
    t_zero_d = Task(task_id=2, vehicle_id="v", size_rho=3.0e6, cpu_phi=10e6, max_delay_d=0.0)
    p_zero_d = compute_task_priority(t_zero_d, dwell_time=10.0, alpha=0.3, beta=0.7)
    expected_dwell_term = 0.3 * math.exp(-1.0 / 10.0)
    assert pytest.approx(p_zero_d, rel=1e-5) == expected_dwell_term

