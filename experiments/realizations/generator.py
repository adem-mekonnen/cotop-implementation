"""
experiments/realizations/generator.py

Deterministic Generator for Controlled Experiment Realizations (Stage 7).
"""

import time
import os
import math
import hashlib
import yaml
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from experiments.realizations.schema import (
    ExperimentRealization,
    TaskRealization,
    VehicleTrajectoryRealization,
    MobilityStateRealization,
    RSUConfigRealization,
    InitialConditionsRealization,
    WorkloadConfigRealization,
    EnvConfigRealization
)


def compute_file_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return ""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class RealizationGenerator:
    def __init__(self, config_path: str = "configs/paper_parameters.yaml"):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.raw_config = yaml.safe_load(f)
            
    def _compute_env_fingerprint(self, geometry: str, obs_dim: int, act_dim: int) -> str:
        payload = {
            "geometry": str(geometry),
            "obs_dim": int(obs_dim),
            "act_dim": int(act_dim),
            "config": self.raw_config
        }
        import json
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def generate_realization(
        self,
        geometry: str,
        workload: int,
        seed: int,
        eval_seed_offset: int = 30000,
        num_vehicles: int = 10
    ) -> ExperimentRealization:
        eval_seed = eval_seed_offset + seed
        rng = np.random.RandomState(eval_seed)
        
        sim_geom = "grid_200m" if geometry in ["grid_200m", "urban_manhattan"] else "corridor_2400m"
        map_scale = 200.0 if sim_geom == "grid_200m" else 2400.0
        
        # 1. RSU Configuration
        num_rsus = 6
        rsu_configs = []
        if sim_geom == "grid_200m":
            rsu_locs = [
                (50.0, 50.0), (100.0, 50.0), (150.0, 50.0),
                (50.0, 150.0), (100.0, 150.0), (150.0, 150.0)
            ]
        else:
            rsu_locs = [(200.0 + i * 400.0, 0.0) for i in range(num_rsus)]
            
        for i in range(num_rsus):
            rsu_configs.append({
                "rsu_id": i,
                "location": [float(rsu_locs[i][0]), float(rsu_locs[i][1])],
                "cpu_capacity_f": 2.0e9,
                "initial_queued_cycles": 0.0,
                "transmission_power_P_R": 100.0,
                "comm_range": 400.0,
                "bandwidth_v2r": 50.0e6,
                "bandwidth_r2r": 50.0e6
            })
            
        # 2. Initial Conditions
        active_veh_ids = [f"v_{v}" for v in range(num_vehicles)]
        initial_conditions = {
            "start_sim_time": 0.0,
            "num_vehicles": num_vehicles,
            "active_vehicle_ids": active_veh_ids,
            "initial_rsu_backlog_cycles": {str(i): 0.0 for i in range(num_rsus)}
        }
        
        # 3. Vehicle Trajectories & Mobility States
        vehicle_trajectories = []
        mobility_states = []
        
        for v in range(num_vehicles):
            v_id = f"v_{v}"
            entry_time = float(v * 2.0)
            initial_speed = float(rng.uniform(30.0, 40.0))
            
            if sim_geom == "grid_200m":
                init_x = float(rng.uniform(10.0, 190.0))
                init_y = float(rng.uniform(10.0, 190.0))
            else:
                init_x = float(v * 150.0)
                init_y = 0.0
                
            traj_points = []
            cur_x, cur_y = init_x, init_y
            time_horizon = 60.0
            dt = 1.0
            
            for step_t in range(int(time_horizon / dt)):
                t_stamp = entry_time + step_t * dt
                if sim_geom == "grid_200m":
                    cur_x = float(np.clip(cur_x + initial_speed * dt * 0.5 * math.cos(v + step_t * 0.1), 0.0, 200.0))
                    cur_y = float(np.clip(cur_y + initial_speed * dt * 0.5 * math.sin(v + step_t * 0.1), 0.0, 200.0))
                else:
                    cur_x = float(np.clip(cur_x + initial_speed * dt, 0.0, 2400.0))
                    cur_y = 0.0
                    
                traj_points.append({
                    "timestamp": round(t_stamp, 2),
                    "x": round(cur_x, 4),
                    "y": round(cur_y, 4),
                    "speed": round(initial_speed, 2)
                })
                
            vehicle_trajectories.append({
                "vehicle_id": v_id,
                "entry_time": round(entry_time, 2),
                "initial_position": [round(init_x, 4), round(init_y, 4)],
                "initial_speed": round(initial_speed, 2),
                "trajectory_points": traj_points
            })
            
            dwell_times = {}
            for r_idx, r_cfg in enumerate(rsu_configs):
                r_loc = r_cfg["location"]
                dist_init = math.sqrt((init_x - r_loc[0])**2 + (init_y - r_loc[1])**2)
                if dist_init <= 400.0:
                    d_rem = max(400.0 - dist_init, 10.0)
                    t_stay = float(d_rem / max(initial_speed, 1.0))
                else:
                    t_stay = float(rng.uniform(5.0, 15.0))
                dwell_times[str(r_idx)] = round(t_stay, 4)
                
            neighbors = [f"v_{other_v}" for other_v in range(num_vehicles) if other_v != v and abs(other_v - v) <= 2]
            mobility_states.append({
                "vehicle_id": v_id,
                "predicted_dwell_time_per_rsu": dwell_times,
                "spatial_proximity_neighbors": neighbors
            })
            
        # 4. Tasks (timestamps and characteristics)
        total_tasks = workload * num_vehicles
        tasks = []
        for i in range(total_tasks):
            v_id = f"v_{i % num_vehicles}"
            v_idx = i % num_vehicles
            v_entry = v_idx * 2.0
            task_time = float(v_entry + (i // num_vehicles) * 0.8)
            
            size_rho = float(rng.uniform(2.0e6, 5.0e6))
            cpu_phi = float(rng.uniform(1.0e6, 10.0e6))
            max_delay_d = float(rng.uniform(20.0, 30.0))
            priority_weight = float(rng.uniform(0.1, 1.0))
            
            tasks.append({
                "task_id": i,
                "vehicle_id": v_id,
                "generation_timestamp": round(task_time, 4),
                "size_rho": round(size_rho, 2),
                "cpu_phi": round(cpu_phi, 2),
                "max_delay_d": round(max_delay_d, 4),
                "priority_weight": round(priority_weight, 6)
            })
            
        workload_config = {
            "tasks_per_vehicle": workload,
            "total_tasks": total_tasks,
            "task_size_range": [2.0e6, 5.0e6],
            "task_deadline_range": [20.0, 30.0],
            "max_task_cpu": 10.0
        }
        
        obs_dim = 4 + (workload * 4) + (num_rsus * 5)
        act_dim = num_rsus + 1
        env_fp = self._compute_env_fingerprint(geometry, obs_dim, act_dim)
        
        env_config = {
            "env_fingerprint": env_fp,
            "comm_model_sha256": compute_file_sha256("envs/comm_model.py"),
            "comp_model_sha256": compute_file_sha256("envs/comp_model.py"),
            "tx_power_vehicle": 0.01,
            "tx_power_rsu": 100.0,
            "noise_power": 0.001,
            "fixed_loss_k": 1000.0,
            "path_loss_factor": 2.0,
            "alpha": 0.3,
            "beta": 0.7,
            "penalty_z": 100.0,
            "epsilon": 0.5
        }
        
        realization_id = f"realization_{geometry}_w{workload}_seed{seed}"
        
        realization = ExperimentRealization(
            realization_id=realization_id,
            geometry=geometry,
            workload=workload,
            seed=seed,
            eval_seed=eval_seed,
            tasks=tasks,
            vehicle_trajectories=vehicle_trajectories,
            mobility_states=mobility_states,
            initial_conditions=initial_conditions,
            rsu_configurations=rsu_configs,
            workload_configuration=workload_config,
            environment_configuration=env_config,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        
        realization.realization_hash = realization.compute_hash()
        return realization
