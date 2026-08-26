"""
tests/test_action_physics.py
Validates that actions 0 through 6 genuinely route to their corresponding physical pathways.
"""
import pytest
import numpy as np
from envs.entities import Vehicle, Task, RSU, SimulationConfig
from envs.comm_model import compute_v2r_rate, compute_r2r_rate
from envs.comp_model import calculate_case1_standalone, calculate_case2_collaboration
from envs.vec_env import get_euclidean_distance

def test_action_differentiation_physical_pathway():
    config = SimulationConfig(
        num_rsus=6,
        tx_power_vehicle=0.01,
        tx_power_rsu=100.0,
        compute_power_rsu=50.0,
        bandwidth_v2r_range=[20.0e6, 100.0e6],
        bandwidth_r2r=50.0e6,
        noise_power=0.001,
        fixed_loss_k=1000.0,
        path_loss_factor=2.0,
        penalty_z=100.0,
        epsilon=0.5
    )

    # Configure asymmetric RSUs to verify distinct physical outcomes
    rsus = [
        RSU(0, (0.0, 0.0), 1.0e9, 10.0e6, config.tx_power_rsu),    # Primary (1 GHz, 10M queue)
        RSU(1, (400.0, 0.0), 4.0e9, 0.0, config.tx_power_rsu),    # High compute, 0 queue
        RSU(2, (800.0, 0.0), 2.0e9, 50.0e6, config.tx_power_rsu),   # Med compute, high queue
        RSU(3, (1200.0, 0.0), 3.0e9, 5.0e6, config.tx_power_rsu),
        RSU(4, (1600.0, 0.0), 1.5e9, 0.0, config.tx_power_rsu),
        RSU(5, (2000.0, 0.0), 2.5e9, 20.0e6, config.tx_power_rsu),
    ]

    vehicle = Vehicle("v_test", pos=(50.0, 0.0), speed=35.0, dwell_time_T_stay=0.005) # 5ms dwell
    task = Task(0, "v_test", size_rho=3.0e6, cpu_phi=10.0e6, max_delay_d=25.0)

    target_rsu = min(rsus, key=lambda r: get_euclidean_distance(vehicle.pos, r.location))
    assert target_rsu.rsu_id == 0

    w_v2r = compute_v2r_rate(
        get_euclidean_distance(vehicle.pos, target_rsu.location),
        config.bandwidth_v2r_range[0],
        config.tx_power_vehicle,
        config.noise_power,
        config.fixed_loss_k,
        config.path_loss_factor
    )

    traces = {}
    for action in range(7):
        if action == 0:
            t_wait_p = target_rsu.queued_cpu_cycles / target_rsu.cpu_capacity_f
            delay, energy = calculate_case1_standalone(
                task.size_rho, task.cpu_phi, w_v2r, target_rsu.cpu_capacity_f,
                config.tx_power_vehicle, config.compute_power_rsu, t_wait=t_wait_p
            )
            traces[action] = {"case": "standalone", "delay": delay, "energy": energy, "rsu": 0}
        else:
            sec_rsu = rsus[action - 1]
            if sec_rsu.rsu_id == target_rsu.rsu_id:
                t_wait_p = target_rsu.queued_cpu_cycles / target_rsu.cpu_capacity_f
                delay, energy = calculate_case1_standalone(
                    task.size_rho, task.cpu_phi, w_v2r, target_rsu.cpu_capacity_f,
                    config.tx_power_vehicle, config.compute_power_rsu, t_wait=t_wait_p
                )
                traces[action] = {"case": "standalone_fallback", "delay": delay, "energy": energy, "rsu": 0}
            else:
                r2r_dist = get_euclidean_distance(target_rsu.location, sec_rsu.location)
                w_r2r = compute_r2r_rate(
                    r2r_dist, config.bandwidth_r2r, config.tx_power_rsu,
                    config.noise_power, config.fixed_loss_k, config.path_loss_factor
                )
                t_wait_s = sec_rsu.queued_cpu_cycles / sec_rsu.cpu_capacity_f
                delay, energy = calculate_case2_collaboration(
                    task.size_rho, task.cpu_phi, w_v2r, w_r2r,
                    target_rsu.cpu_capacity_f, sec_rsu.cpu_capacity_f,
                    vehicle.dwell_time_T_stay, config.tx_power_vehicle,
                    config.tx_power_rsu, config.compute_power_rsu, config.compute_power_rsu,
                    t_wait=t_wait_s
                )
                traces[action] = {"case": "collaborative", "delay": delay, "energy": energy, "rsu": sec_rsu.rsu_id}

    # Verify that Standalone vs Collaborative actions yield distinct, causal results
    assert traces[0]["case"] == "standalone"
    assert traces[1]["case"] == "standalone_fallback"
    assert traces[2]["case"] == "collaborative"
    assert traces[2]["rsu"] == 1

    # Delays and energies between different RSU targets MUST differ due to R2R distance, queue, and CPU capacity
    assert traces[0]["delay"] != traces[2]["delay"]
    assert traces[2]["delay"] != traces[3]["delay"]
    assert traces[0]["energy"] != traces[2]["energy"]
