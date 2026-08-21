vec_env_code = '''
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import os
from typing import Tuple

from envs.entities import Vehicle, Task, RSU, SimulationConfig
from envs.state_builder import build_state
from envs.comm_model import compute_v2r_rate, compute_r2r_rate, get_euclidean_distance
from envs.comp_model import calculate_case1_standalone, calculate_case2_collaboration
from envs.task_generator import TaskGenerator
from utils.task_priority import prioritize_tasks
from envs.sumo_manager import SumoManager

MOBILITY_CHECKPOINT = "results/checkpoints/mobility_model.pth"
TRAJ_HISTORY_LEN = 5


class VECEnv(gym.Env):
    def __init__(self, config: SimulationConfig, port: int = None, use_mobility_model: bool = True, use_priority: bool = True):
        super(VECEnv, self).__init__()
        self.config = config
        self.port = port
        self.use_mobility_model = use_mobility_model
        self.use_priority = use_priority

        self.action_space = spaces.Discrete(self.config.num_rsus)
        n_tasks = self.config.num_tasks_per_vehicle_range[0]
        self.obs_dim = 4 + (n_tasks * 4) + (self.config.num_rsus * 5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.sumo_manager = SumoManager("sumo_config/hangzhou.sumocfg", port=self.port, use_gui=False)
        self.sumo_started = False
        self.current_vehicle = None
        self.current_tasks = []
        self.rsus = []
        self.current_task_idx = 0
        self.task_gen = TaskGenerator(self.config)

        self.mobility_model = None
        if self.use_mobility_model and os.path.exists(MOBILITY_CHECKPOINT):
            try:
                from models.mobility_gat import MobilityGAT_GRU
                self.mobility_model = MobilityGAT_GRU(
                    input_dim=2, embed_dim=64, num_heads=4, gru_hidden=64, output_dim=2
                )
                self.mobility_model.load_state_dict(torch.load(MOBILITY_CHECKPOINT, map_location='cpu'))
                self.mobility_model.eval()
                print(f'Mobility model loaded from {MOBILITY_CHECKPOINT} (port={self.port})')
            except Exception as e:
                print(f'Could not load mobility model ({e}); using distance-based fallback.')
                self.mobility_model = None

    def _estimate_dwell_time(self, vehicle: Vehicle, current_rsu: RSU) -> float:
        if self.mobility_model is not None and len(vehicle.trajectory_history) >= TRAJ_HISTORY_LEN:
            try:
                traj = np.array(vehicle.trajectory_history[-TRAJ_HISTORY_LEN:], dtype=np.float32)
                x_seq = torch.FloatTensor(traj).unsqueeze(0)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                with torch.no_grad():
                    predictions = self.mobility_model(x_seq, edge_index)
                future_pos = predictions[0, -1].numpy()
                dist_to_edge = self.config.rsu_comm_range - get_euclidean_distance(
                    (float(future_pos[0]), float(future_pos[1])), current_rsu.location
                )
                speed = max(vehicle.speed, 1e-3)
                return max(dist_to_edge / speed, 0.5)
            except Exception:
                pass

        dist_to_rsu = get_euclidean_distance(vehicle.pos, current_rsu.location)
        remaining = max(self.config.rsu_comm_range - dist_to_rsu, 0.1)
        speed = max(vehicle.speed, 1e-3)
        return remaining / speed

    def _get_obs(self):
        return build_state(self.current_vehicle, self.current_tasks, self.rsus, self.config)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.current_vehicle is None or self.current_task_idx >= len(self.current_tasks):
            return self._get_obs(), 0.0, True, False, {}

        task = self.current_tasks[self.current_task_idx]
        target_rsu = self.rsus[action]

        w_v2r = compute_v2r_rate(
            pos_v=self.current_vehicle.pos,
            pos_r=target_rsu.location,
            bandwidth_B=self.config.bandwidth_v2r_range[0],
            power_P_V=self.config.tx_power_vehicle,
            noise_power=self.config.noise_power,
            fixed_loss_k=self.config.fixed_loss_k,
            path_loss_factor=self.config.path_loss_factor
        )

        dwell_time = self._estimate_dwell_time(self.current_vehicle, target_rsu)
        self.current_vehicle.dwell_time_T_stay = dwell_time

        standalone_delay, energy = calculate_case1_standalone(
            task_size_rho=task.size_rho,
            task_cpu_phi=task.cpu_phi,
            w_v2r=w_v2r,
            rsu_cpu_f=target_rsu.cpu_capacity_f,
            power_v=self.config.tx_power_vehicle,
            power_rsu=target_rsu.transmission_power_P_R,
        )

        if target_rsu.cpu_capacity_f > 0:
            standalone_delay += target_rsu.queue_length / target_rsu.cpu_capacity_f

        case_used = 1
        if dwell_time < standalone_delay:
            other_rsus = [r for r in self.rsus if r.rsu_id != target_rsu.rsu_id]
            next_rsu = min(other_rsus, key=lambda r: get_euclidean_distance(target_rsu.location, r.location))

            w_r2r = compute_r2r_rate(
                pos_r1=target_rsu.location,
                pos_r2=next_rsu.location,
                bandwidth_B=self.config.bandwidth_r2r,
                power_P_R=self.config.tx_power_rsu,
                noise_power=self.config.noise_power,
                fixed_loss_k=self.config.fixed_loss_k,
                path_loss_factor=self.config.path_loss_factor
            )

            standalone_delay, energy = calculate_case2_collaboration(
                task_size_rho=task.size_rho,
                task_cpu_phi=task.cpu_phi,
                w_v2r=w_v2r,
                w_r2r=w_r2r,
                rsu1_cpu_f=target_rsu.cpu_capacity_f,
                rsu2_cpu_f=next_rsu.cpu_capacity_f,
                t1_dwell_time=dwell_time,
                power_v=self.config.tx_power_vehicle,
                power_rsu1=target_rsu.transmission_power_P_R,
                power_rsu2=next_rsu.transmission_power_P_R,
            )
            case_used = 2

        target_rsu.queue_length += 1

        eps = getattr(self.config, 'epsilon', 0.5)
        if standalone_delay > task.max_delay_d:
            reward = -self.config.penalty_z
        else:
            reward = -(eps * standalone_delay + (1 - eps) * energy)

        self.current_task_idx += 1
        terminated = (self.current_task_idx >= len(self.current_tasks))

        if target_rsu.queue_length > 0:
            target_rsu.queue_length -= 1

        info = {"delay": standalone_delay, "energy": energy, "case": case_used}
        return self._get_obs(), float(reward), terminated, False, info

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if not self.sumo_started:
            self.sumo_manager.start_simulation()
            self.sumo_started = True
        else:
            self.sumo_manager.reload_simulation()

        vehicles_data = {}
        while not vehicles_data:
            self.sumo_manager.step()
            vehicles_data = self.sumo_manager.get_vehicle_data()

        v_id = list(vehicles_data.keys())[0]
        v_data = vehicles_data[v_id]

        prev_history = []
        if self.current_vehicle is not None and self.current_vehicle.v_id == v_id:
            prev_history = self.current_vehicle.trajectory_history

        new_history = (prev_history + [v_data['pos']])[-TRAJ_HISTORY_LEN:]

        self.current_vehicle = Vehicle(
            v_id=v_id, pos=v_data['pos'], speed=v_data['speed'],
            dwell_time_T_stay=0.0, trajectory_history=new_history
        )

        self.current_tasks = self.task_gen.generate_tasks_for_vehicle(v_id)

        if not self.rsus:
            positions = [(0, 0), (100, 0), (200, 0), (0, 100), (100, 100), (200, 100)]
            self.rsus = [
                RSU(i, positions[i], self.config.rsu_cpu_capacity_range[0], 0, self.config.tx_power_rsu)
                for i in range(self.config.num_rsus)
            ]

        nearest_rsu = min(self.rsus, key=lambda r: get_euclidean_distance(self.current_vehicle.pos, r.location))
        dwell_estimate = self._estimate_dwell_time(self.current_vehicle, nearest_rsu)
        self.current_vehicle.dwell_time_T_stay = dwell_estimate

        if self.use_priority:
            self.current_tasks = prioritize_tasks(
                self.current_tasks, dwell_estimate, self.config.alpha, self.config.beta
            )

        self.current_task_idx = 0
        return self._get_obs(), {}

    def close(self):
        if self.sumo_manager:
            self.sumo_manager.close()
'''
with open('envs/vec_env.py', 'w') as f:
    f.write(vec_env_code.strip())
print("Final vec_env.py: comm_model and comp_model calls corrected to match actual function signatures.")