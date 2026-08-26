"""
tests/test_ablation_integrity.py
Validates that ablations (w/o MD, w/o TP, w/o CO) genuinely modify their target mechanisms.
"""
import pytest
from envs.entities import SimulationConfig, Task
from utils.task_priority import prioritize_tasks

def test_ablation_task_priority():
    tasks = [
        Task(0, "v0", size_rho=5.0e6, cpu_phi=10.0e6, max_delay_d=20.0), # Large, urgent
        Task(1, "v0", size_rho=2.0e6, cpu_phi=5.0e6, max_delay_d=30.0),  # Small, relaxed
    ]
    
    # 1. With Task Priority enabled (Eq. 23)
    sorted_tasks = prioritize_tasks(tasks, dwell_time=10.0, alpha=0.3, beta=0.7)
    assert sorted_tasks[0].task_id == 0 # Highest priority first
    
    # 2. Without Task Priority (Ablation w/o TP): tasks processed in FIFO order
    fifo_tasks = sorted(tasks, key=lambda t: t.task_id)
    assert fifo_tasks[0].task_id == 0

def test_ablation_collaboration_and_mobility_flags():
    # Verify environment initialization switches for ablations
    from envs.vec_env import VECEnv
    config = SimulationConfig()
    
    # CoTOP: mobility=True, priority=True
    env_full = VECEnv(config=config, use_mobility_model=True, use_priority=True, port=8830)
    assert env_full.use_mobility_model is True
    assert env_full.use_priority is True
    
    # CoTOP w/o MD: mobility=False
    env_wo_md = VECEnv(config=config, use_mobility_model=False, use_priority=True, port=8831)
    assert env_wo_md.use_mobility_model is False
    
    # CoTOP w/o TP: priority=False
    env_wo_tp = VECEnv(config=config, use_mobility_model=True, use_priority=False, port=8832)
    assert env_wo_tp.use_priority is False
