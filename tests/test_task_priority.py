import pytest
from envs.entities import Task
from utils.task_priority import compute_task_priority, prioritize_tasks

def test_task_priority_ordering():
    # Stricter deadline -> higher priority
    # Larger data volume -> higher priority
    # Longer dwell time -> higher priority
    t1 = Task(task_id=1, vehicle_id="v1", size_rho=5.0e6, cpu_phi=10.0e6, max_delay_d=10.0) # High size, low deadline
    t2 = Task(task_id=2, vehicle_id="v1", size_rho=2.0e6, cpu_phi=10.0e6, max_delay_d=30.0) # Low size, high deadline
    
    dwell_time = 15.0
    p1 = compute_task_priority(t1, dwell_time, alpha=0.3, beta=0.7)
    p2 = compute_task_priority(t2, dwell_time, alpha=0.3, beta=0.7)
    assert p1 > p2
    
    sorted_tasks = prioritize_tasks([t2, t1], dwell_time, alpha=0.3, beta=0.7)
    assert sorted_tasks[0].task_id == 1
    assert sorted_tasks[1].task_id == 2
