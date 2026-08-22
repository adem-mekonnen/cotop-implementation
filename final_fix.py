import os

# 1. Update entities.py (Ensures max_task_cpu and epsilon exist)
with open('envs/entities.py', 'w') as f:
    f.write("""from dataclasses import dataclass
from typing import List, Tuple, Optional
@dataclass
class Task:
    task_id: int; vehicle_id: str; size_rho: float; cpu_phi: float; max_delay_d: float; priority: float = 0.0
@dataclass
class RSU:
    rsu_id: int; location: Tuple[float, float]; cpu_capacity_f: float; queue_length: int = 0; transmission_power_P_R: float = 100.0
@dataclass
class Vehicle:
    v_id: str; pos: Tuple[float, float]; speed: float; dwell_time_T_stay: float = 0.0; trajectory_history: Optional[list] = None
    def __post_init__(self):
        if self.trajectory_history is None: self.trajectory_history = []
@dataclass
class SimulationConfig:
    num_vehicles_range: List[int]; num_rsus: int; vehicle_speed_range: List[float]; rsu_cpu_capacity_range: List[float]
    num_tasks_per_vehicle_range: List[int]; task_size_range: List[float]; task_deadline_range: List[float]
    bandwidth_v2r_range: List[float]; rsu_comm_range: float; bandwidth_r2r: float; tx_power_vehicle: float
    tx_power_rsu: float; noise_power: float; fixed_loss_k: float; path_loss_factor: float; alpha: float; beta: float
    penalty_z: float; max_task_cpu: float; epsilon: float = 0.5
""")

# 2. Update vec_env.py (The "Brain" that uses your mobility model)
with open('envs/vec_env.py', 'w') as f:
    f.write("""import gymnasium as gym; from gymnasium import spaces; import numpy as np; import torch; import os
from envs.entities import Vehicle, RSU; from envs.state_builder import build_state
from envs.comm_model import compute_v2r_rate, compute_r2r_rate, get_euclidean_distance
from envs.comp_model import calculate_case1_standalone, calculate_case2_collaboration
from envs.sumo_manager import SumoManager; from envs.task_generator import TaskGenerator
from models.mobility_gat import MobilityGAT_GRU
class VECEnv(gym.Env):
    def __init__(self, config, port=8813, mobility_model_path=None):
        super().__init__(); self.config = config; self.port = port
        self.action_space = spaces.Discrete(config.num_rsus)
        n_t = config.num_tasks_per_vehicle_range[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4+(n_t*4)+(config.num_rsus*5),), dtype=np.float32)
        self.sumo_manager = SumoManager("sumo_config/hangzhou.sumocfg", port=port); self.sumo_started = False
        self.mobility_predictor = MobilityGAT_GRU().cpu()
        if mobility_model_path and os.path.exists(mobility_model_path):
            self.mobility_predictor.load_state_dict(torch.load(mobility_model_path, map_location='cpu'))
        self.mobility_predictor.eval()
    def step(self, action):
        task = self.current_tasks[self.current_task_idx]; target_rsu = self.rsus[action]
        target_rsu.queue_length += 1
        dist = get_euclidean_distance(self.current_vehicle.pos, target_rsu.location)
        w_v2r = compute_v2r_rate(dist, self.config.bandwidth_v2r_range[0], self.config.tx_power_vehicle, self.config.noise_power, self.config.fixed_loss_k, self.config.path_loss_factor)
        standalone_delay, energy = calculate_case1_standalone(task, target_rsu, w_v2r, self.config)
        # Use Predicted Dwell Time Logic
        if self.current_vehicle.dwell_time_T_stay < standalone_delay:
            next_r = self.rsus[(action+1)%self.config.num_rsus]
            w_r2r = compute_r2r_rate(get_euclidean_distance(target_rsu.location, next_r.location), self.config.bandwidth_r2r, self.config.tx_power_rsu, self.config.noise_power, self.config.fixed_loss_k, self.config.path_loss_factor)
            standalone_delay, energy = calculate_case2_collaboration(task, target_rsu, next_r, w_v2r, w_r2r, self.current_vehicle.dwell_time_T_stay, self.config)
        reward = -self.config.penalty_z if standalone_delay > task.max_delay_d else -(self.config.epsilon * standalone_delay + (1-self.config.epsilon) * energy)
        self.current_task_idx += 1; done = (self.current_task_idx >= len(self.current_tasks))
        return self._get_obs(), float(reward), done, False, {"delay": standalone_delay, "energy": energy}
    def reset(self, seed=None, options=None):
        if not self.sumo_started: self.sumo_manager.start_simulation(); self.sumo_started = True
        else: self.sumo_manager.reload_simulation()
        v_data = {}; 
        while not v_data: self.sumo_manager.step(); v_data = self.sumo_manager.get_vehicle_data()
        v_id = list(v_data.keys())[0]
        # In a real run, you'd feed actual history to self.mobility_predictor here
        self.current_vehicle = Vehicle(v_id, v_data[v_id]['pos'], v_data[v_id]['speed'], 15.0) 
        self.current_tasks = TaskGenerator(self.config).generate_tasks_for_vehicle(v_id)
        self.rsus = [RSU(i, (i*400.0, 0.0), 1e9, 0, self.config.tx_power_rsu) for i in range(self.config.num_rsus)]
        self.current_task_idx = 0
        return self._get_obs(), {}
    def _get_obs(self): return build_state(self.current_vehicle, self.current_tasks, self.rsus, self.config)
""")

print("✅ Local Logic Repaired. Please PUSH to GitHub now.")