"""
experiments/action_physics_debug.py
Structured debug script demonstrating end-to-end action differentiation across actions 0 to 6.
"""
import math
import numpy as np
import pandas as pd
from envs.entities import Vehicle, Task, RSU, SimulationConfig
from envs.comm_model import compute_v2r_rate, compute_r2r_rate
from envs.comp_model import calculate_case1_standalone, calculate_case2_collaboration
from envs.vec_env import get_euclidean_distance

def run_action_debug():
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

    rsus = [
        RSU(0, (0.0, 0.0), 1.0e9, 10.0e6, config.tx_power_rsu),
        RSU(1, (400.0, 0.0), 4.0e9, 0.0, config.tx_power_rsu),
        RSU(2, (800.0, 0.0), 2.0e9, 30.0e6, config.tx_power_rsu),
        RSU(3, (1200.0, 0.0), 3.0e9, 5.0e6, config.tx_power_rsu),
        RSU(4, (1600.0, 0.0), 1.5e9, 0.0, config.tx_power_rsu),
        RSU(5, (2000.0, 0.0), 2.5e9, 15.0e6, config.tx_power_rsu),
    ]

    vehicle = Vehicle("v_debug", pos=(80.0, 0.0), speed=35.0, dwell_time_T_stay=0.01)
    task = Task(0, "v_debug", size_rho=4.0e6, cpu_phi=10.0e6, max_delay_d=25.0)

    target_rsu = min(rsus, key=lambda r: get_euclidean_distance(vehicle.pos, r.location))
    w_v2r = compute_v2r_rate(
        get_euclidean_distance(vehicle.pos, target_rsu.location),
        20.0e6, config.tx_power_vehicle, config.noise_power,
        config.fixed_loss_k, config.path_loss_factor
    )

    records = []
    for action in range(7):
        if action == 0:
            t_wait_p = target_rsu.queued_cpu_cycles / target_rsu.cpu_capacity_f
            delay, energy = calculate_case1_standalone(
                task.size_rho, task.cpu_phi, w_v2r, target_rsu.cpu_capacity_f,
                config.tx_power_vehicle, config.compute_power_rsu, t_wait=t_wait_p
            )
            mode = "Case 1: Standalone"
            target_desc = f"Primary RSU {target_rsu.rsu_id}"
            w_r2r_val = 0.0
        else:
            sec_rsu = rsus[action - 1]
            if sec_rsu.rsu_id == target_rsu.rsu_id:
                t_wait_p = target_rsu.queued_cpu_cycles / target_rsu.cpu_capacity_f
                delay, energy = calculate_case1_standalone(
                    task.size_rho, task.cpu_phi, w_v2r, target_rsu.cpu_capacity_f,
                    config.tx_power_vehicle, config.compute_power_rsu, t_wait=t_wait_p
                )
                mode = "Case 1: Fallback"
                target_desc = f"Primary RSU {target_rsu.rsu_id}"
                w_r2r_val = 0.0
            else:
                r2r_dist = get_euclidean_distance(target_rsu.location, sec_rsu.location)
                w_r2r = compute_r2r_rate(
                    r2r_dist, config.bandwidth_r2r, config.tx_power_rsu,
                    config.noise_power, config.fixed_loss_k, config.path_loss_factor
                )
                w_r2r_val = w_r2r
                t_wait_s = sec_rsu.queued_cpu_cycles / sec_rsu.cpu_capacity_f
                delay, energy = calculate_case2_collaboration(
                    task.size_rho, task.cpu_phi, w_v2r, w_r2r,
                    target_rsu.cpu_capacity_f, sec_rsu.cpu_capacity_f,
                    vehicle.dwell_time_T_stay, config.tx_power_vehicle,
                    config.tx_power_rsu, config.compute_power_rsu, config.compute_power_rsu,
                    t_wait=t_wait_s
                )
                mode = "Case 2: Collaborative"
                target_desc = f"Sec RSU {sec_rsu.rsu_id} (D={r2r_dist:.0f}m)"

        reward = -(0.5 * delay + 0.5 * energy) if delay <= task.max_delay_d else -config.penalty_z
        records.append({
            "Action": action,
            "Target": target_desc,
            "Mode": mode,
            "V2R Rate (Mbps)": round(w_v2r / 1e6, 2),
            "R2R Rate (Mbps)": round(w_r2r_val / 1e6, 2),
            "Delay (s)": round(delay, 4),
            "Energy (J)": round(energy, 4),
            "Reward": round(reward, 4),
        })

    df = pd.DataFrame(records)
    print("\n=== ACTION-TO-PHYSICS DIFFERENTIATION MATRIX ===")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    run_action_debug()
