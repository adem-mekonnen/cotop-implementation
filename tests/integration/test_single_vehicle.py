import pytest
import math
from envs.entities import Vehicle, Task, RSU, SimulationConfig
from envs.comm_model import compute_v2r_rate, compute_r2r_rate
from envs.comp_model import calculate_case1_standalone, calculate_case2_collaboration
from utils.task_priority import prioritize_tasks

def test_single_vehicle_end_to_end_pipeline():
    """
    Validates the end-to-end mathematical pipeline:
    Vehicle -> Task Prioritization -> Primary RSU selection -> Transmission & Processing -> Delay & Energy
    """
    vehicle = Vehicle(v_id="veh_0", pos=(100.0, 0.0), speed=30.0, dwell_time_T_stay=10.0)
    rsus = [
        RSU(rsu_id=0, location=(0.0, 0.0), cpu_capacity_f=2.0e9, queued_cpu_cycles=0.0, transmission_power_P_R=100.0),
        RSU(rsu_id=1, location=(400.0, 0.0), cpu_capacity_f=2.0e9, queued_cpu_cycles=0.0, transmission_power_P_R=100.0),
    ]
    tasks = [
        Task(task_id=0, vehicle_id="veh_0", size_rho=2.0e6, cpu_phi=10.0e6, max_delay_d=25.0),
        Task(task_id=1, vehicle_id="veh_0", size_rho=4.0e6, cpu_phi=20.0e6, max_delay_d=15.0),
    ]
    
    # 1. Prioritize tasks
    sorted_tasks = prioritize_tasks(tasks, vehicle.dwell_time_T_stay, alpha=0.3, beta=0.7)
    assert sorted_tasks[0].task_id == 1  # Larger size and shorter deadline has higher priority
    
    # 2. Nearest RSU is RSU 0 (dist=100m vs dist=300m)
    primary_rsu = rsus[0]
    dist_v2r = 100.0
    w_v2r = compute_v2r_rate(
        distance=dist_v2r,
        bandwidth_B=20.0e6,
        power_P_V=0.01,
        noise_power=0.001,
        fixed_loss_k=1000.0,
        path_loss_factor=2.0
    )
    assert pytest.approx(w_v2r, rel=1e-5) == 20.0e6
    
    # 3. Process Task 1 standalone (Case 1)
    t = sorted_tasks[0]
    delay, energy = calculate_case1_standalone(
        task_size_rho=t.size_rho,
        task_cpu_phi=t.cpu_phi,
        w_v2r=w_v2r,
        rsu_cpu_f=primary_rsu.cpu_capacity_f,
        power_v=0.01,
        compute_power_rsu=50.0,
        t_wait=0.0
    )
    # size_bits = 4e6 * 8 = 32e6 bits -> t_trans = 32e6 / 20e6 = 1.6s
    # t_comp = 20e6 / 2e9 = 0.01s
    # delay = 1.61s
    # energy = (0.01 * 1.6) + (0.01 * 50) = 0.016 + 0.5 = 0.516 J
    assert pytest.approx(delay, rel=1e-5) == 1.61
    assert pytest.approx(energy, rel=1e-5) == 0.516
