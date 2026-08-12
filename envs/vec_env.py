import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple, Dict, Any

from envs.entities import Vehicle, Task, RSU, SimulationConfig
from envs.state_builder import build_state
from envs.comm_model import compute_v2r_rate, compute_r2r_rate, get_euclidean_distance
from envs.comp_model import calculate_case1_standalone, calculate_case2_collaboration

class VECEnv(gym.Env):
    def __init__(self, config: SimulationConfig):
        super(VECEnv, self).__init__()
        
        self.config = config
        
        # Action space: select one of M RSUs (0 to M-1)
        self.action_space = spaces.Discrete(self.config.num_rsus)
        
        # Ensure observation space calculation is identical to build_state logic
        # Vehicle(4) + Tasks(I * 4) + RSUs(M * 5)
        self.obs_dim = 4 + (self.config.num_tasks_per_vehicle * 4) + (self.config.num_rsus * 5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        self.sumo_manager = None 
        self.current_vehicle = None
        self.current_tasks = []
        self.rsus = []
        self.current_task_idx = 0

    def _get_obs(self):
        # Always use build_state to ensure consistency
        return build_state(self.current_vehicle, self.current_tasks, self.rsus, self.obs_dim)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.current_vehicle is None or self.current_task_idx >= len(self.current_tasks):
            return self._get_obs(), 0.0, True, False, {}
            
        task = self.current_tasks[self.current_task_idx]
        target_rsu = self.rsus[action]
        
        # --- 1. Communication Rates (Eq 1-2) ---
        dist_v_r = get_euclidean_distance(self.current_vehicle.pos, target_rsu.location)
        w_v2r = compute_v2r_rate(
            distance=max(dist_v_r, 1.0), # Distance guard
            bandwidth=self.config.bandwidth_v2r_range[0],
            tx_power=self.config.tx_power_vehicle,
            noise_power=self.config.noise_power,
            fixed_loss=self.config.fixed_loss_k,
            sigma=self.config.path_loss_factor
        )

        # --- 2. Handoff Logic (Sec III-C) ---
        t1_dwell = self.current_vehicle.dwell_time_T_stay
        # Rough estimate for standalone
        standalone_delay, _ = calculate_case1_standalone(task, target_rsu, w_v2r, self.config)
        
        if t1_dwell >= standalone_delay:
            delay, energy = calculate_case1_standalone(task, target_rsu, w_v2r, self.config)
            case_used = 1
        else:
            # Collaboration logic (Eq 7-10)
            next_rsu = self.rsus[(action + 1) % self.config.num_rsus]
            dist_r_r = get_euclidean_distance(target_rsu.location, next_rsu.location)
            w_r2r = compute_r2r_rate(dist_r_r, self.config.bandwidth_r2r, self.config.tx_power_rsu, 
                                     self.config.noise_power, self.config.fixed_loss_k, self.config.path_loss_factor)
            delay, energy = calculate_case2_collaboration(task, target_rsu, next_rsu, w_v2r, w_r2r, t1_dwell, self.config)
            case_used = 2
        
        # --- 3. Reward (Eq 25) ---
        if delay > task.max_delay_d:
            reward = -self.config.penalty_z
        else:
            reward = -(self.config.alpha * delay + self.config.beta * energy)
            
        self.current_task_idx += 1
        terminated = (self.current_task_idx >= len(self.current_tasks))
        truncated = False
        
        info = {"delay": delay, "energy": energy, "case": case_used}
        return self._get_obs(), float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        if self.sumo_manager is None:
            from envs.sumo_manager import SumoManager
            self.sumo_manager = SumoManager("sumo_config/hangzhou.sumocfg", use_gui=False)
            self.sumo_manager.start_simulation()
            
        # 1. Spawn Vehicle (Safety Loop)
        vehicles_data = {}
        while not vehicles_data:
            self.sumo_manager.step()
            vehicles_data = self.sumo_manager.get_vehicle_data()
            
        v_id = list(vehicles_data.keys())[0]
        v_data = vehicles_data[v_id]
        
        # 2. Assign Vehicle & Tasks
        self.current_vehicle = Vehicle(v_id=v_id, pos=v_data['pos'], speed=v_data['speed'],
                                       dwell_time_T_stay=v_data.get('predicted_dwell', 20.0))
        
        from envs.task_generator import TaskGenerator
        self.current_tasks = TaskGenerator(self.config).generate_tasks_for_vehicle(v_id)
        
        # 3. Dynamic RSU Placement (aligned with Table III)
        if not self.rsus:
            # Place RSUs along a 2000m path (adjust to your map size)
            self.rsus = [
                RSU(i, (i * 400.0, 0), self.config.rsu_cpu_capacity_range[0], 0, self.config.tx_power_rsu) 
                for i in range(self.config.num_rsus)
            ]
            
        self.current_task_idx = 0
        return self._get_obs(), {}

    def close(self):
        if self.sumo_manager:
            self.sumo_manager.close_simulation()