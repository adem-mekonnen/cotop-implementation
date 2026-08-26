import pytest
import numpy as np
from envs.entities import Vehicle, Task, RSU, SimulationConfig
from envs.state_builder import build_state
import yaml

def test_state_builder_dimensions_and_normalization():
    with open("configs/paper_parameters.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    config = SimulationConfig(**cfg_dict)
    
    vehicle = Vehicle(v_id="v1", pos=(1200.0, 100.0), speed=35.0, dwell_time_T_stay=10.0)
    tasks = [
        Task(task_id=i, vehicle_id="v1", size_rho=3.0e6, cpu_phi=8.0e6, max_delay_d=25.0, priority=0.5)
        for i in range(config.num_tasks_per_vehicle_range[0])
    ]
    rsus = [
        RSU(rsu_id=i, location=(i * 400.0, 0.0), cpu_capacity_f=2.0e9, queued_cpu_cycles=1.0e7, transmission_power_P_R=100.0)
        for i in range(config.num_rsus)
    ]
    
    state = build_state(vehicle, tasks, rsus, config)
    
    # Expected dim = 4 + (max_tasks * 4) + (num_rsus * 5)
    expected_dim = 4 + (20 * 4) + (6 * 5)
    assert state.shape == (expected_dim,)
    
    # Assert all features are normalized within [0, 1] range
    assert np.all(state >= 0.0)
    assert np.all(state <= 1.0)
