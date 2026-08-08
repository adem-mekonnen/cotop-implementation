import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple, Dict, Any

from envs.entities import Vehicle, Task, RSU, SimulationConfig
from envs.state_builder import build_state
from envs.comm_model import compute_v2r_rate
from envs.comp_model import calculate_case1_standalone

class VECEnv(gym.Env):
    """
    Reinforcement Learning environment for CoTOP (Equation 24-25).
    """
    def __init__(self, config: SimulationConfig, num_rsus: int = 6, max_tasks: int = 3):
        super(VECEnv, self).__init__()
        
        self.config = config
        self.num_rsus = num_rsus
        self.max_tasks = max_tasks
        
        self.action_space = spaces.Discrete(self.num_rsus)
        
        # State dimension: Vehicle(4) + Tasks(max_tasks*4) + RSUs(num_rsus*5)
        obs_dim = 4 + (self.max_tasks * 4) + (self.num_rsus * 5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.sumo_manager = None 
        self.current_vehicle = None
        self.current_tasks = []
        self.rsus = []
        self.current_task_idx = 0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes an offloading decision and calculates reward based on Eq 25.
        Returns: state, reward, terminated, truncated, info
        """
        if self.current_task_idx >= len(self.current_tasks):
            state = build_state(self.current_vehicle, self.current_tasks, self.rsus)
            return state, 0.0, True, False, {}
            
        task = self.current_tasks[self.current_task_idx]
        chosen_rsu = self.rsus[action]
        
        # V2R Data Rate using values from SimulationConfig
        w_v2r = compute_v2r_rate(
            pos_v=self.current_vehicle.pos,
            pos_r=chosen_rsu.location,
            bandwidth_B=60.0e6,
            power_P_V=0.01, 
            noise_power=0.001,
            fixed_loss_k=1000.0,
            path_loss_factor=2.0
        )
        
        delay, energy = calculate_case1_standalone(
            task_size_rho=task.size_rho,
            task_cpu_phi=task.cpu_phi,
            w_v2r=w_v2r,
            rsu_cpu_f=chosen_rsu.cpu_capacity_f,
            power_v=0.01, 
            power_rsu=100.0
        )
        
        # Eq 25: Reward Function
        Z_penalty = 1000.0  
        alpha_weight = 0.3  
        beta_weight = 0.7   
        
        if delay > task.max_delay_d:
            reward = -Z_penalty
        else:
            reward = -(alpha_weight * delay + beta_weight * energy)
            
        self.current_task_idx += 1
        terminated = (self.current_task_idx >= len(self.current_tasks))
        truncated = False
        
        next_state = build_state(self.current_vehicle, self.current_tasks, self.rsus)
        
        info = {
            "delay": delay,
            "energy": energy,
            "missed_deadline": delay > task.max_delay_d
        }
        
        return next_state, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment (Gymnasium API).
        Returns: state, info
        """
        super().reset(seed=seed)
            
        if self.sumo_manager is None:
            from envs.sumo_manager import SumoManager
            # Ensure SUMO config path matches deployed map files
            self.sumo_manager = SumoManager("sumo_config/hangzhou.sumocfg", use_gui=False)
            self.sumo_manager.start_simulation()
            
        # Logic Correction: Advance simulation until at least one vehicle spawns
        vehicles = []
        while not vehicles:
            self.sumo_manager.step()
            vehicles = self.sumo_manager.get_vehicle_data()
            
        self.current_vehicle = vehicles[0]
        
        from envs.task_generator import TaskGenerator
        if not hasattr(self, 'task_generator'):
            self.task_generator = TaskGenerator(self.max_tasks)
        self.current_tasks = self.task_generator.generate_tasks_for_vehicle(self.current_vehicle.v_id)
        
        if not self.rsus:
            # Generate RSUs matching num_rsus
            self.rsus = [RSU(i, (i*100, i*100), 1.0e9, 0, 100.0) for i in range(self.num_rsus)]
            
        self.current_task_idx = 0
        state = build_state(self.current_vehicle, self.current_tasks, self.rsus)
        
        return state, {}

    def render(self):
        pass
