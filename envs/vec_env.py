import gym
from gym import spaces
import numpy as np
from typing import List, Tuple

from envs.entities import Vehicle, Task, RSU, SimulationConfig
from envs.state_builder import build_state
from envs.comm_model import compute_v2r_rate
from envs.comp_model import calculate_case1_standalone, calculate_case2_collaboration

class VECEnv(gym.Env):
    """
    Reinforcement Learning environment for CoTOP (Equation 24-25).
    Integrates SUMO (placeholder), Communication Model, and Computation Model.
    """
    def __init__(self, config: SimulationConfig, num_rsus: int = 5, max_tasks: int = 3):
        super(VECEnv, self).__init__()
        
        self.config = config
        self.num_rsus = num_rsus
        self.max_tasks = max_tasks
        
        # Action space: Choose which RSU to offload the current task to
        self.action_space = spaces.Discrete(self.num_rsus)
        
        # Observation space shape based on build_state
        obs_dim = 4 + (self.max_tasks * 4) + (self.num_rsus * 5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Integration points
        self.sumo_manager = None # Placeholder for SumoManager
        
        # State variables
        self.current_vehicle = None
        self.current_tasks = []
        self.rsus = []
        self.current_task_idx = 0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Executes an offloading decision and calculates reward based on Eq 25.
        """
        if self.current_task_idx >= len(self.current_tasks):
            return build_state(self.current_vehicle, self.current_tasks, self.rsus), 0.0, True, {}
            
        task = self.current_tasks[self.current_task_idx]
        chosen_rsu = self.rsus[action]
        
        # --- Physics Integration ---
        # V2R Data Rate
        w_v2r = compute_v2r_rate(
            pos_v=self.current_vehicle.pos,
            pos_r=chosen_rsu.location,
            bandwidth_B=self.config.bandwidth_B,
            power_P_V=1.0, # Vehicle transmission power (P_V)
            noise_power_sigma2=self.config.noise_power_sigma2
        )
        
        # Delay and Energy calculation
        # Simplifying to Case 1 (Standalone) for the step logic; 
        # Case 2 would be used if action indicated collaboration between RSUs.
        delay, energy = calculate_case1_standalone(
            task_size_rho=task.size_rho,
            task_cpu_phi=task.cpu_phi,
            w_v2r=w_v2r,
            rsu_cpu_f=chosen_rsu.cpu_capacity_f,
            power_v=1.0, 
            power_rsu=chosen_rsu.transmission_power_P_R
        )
        
        # --- Reward Calculation (Eq 25) ---
        Z_penalty = 1000.0  # Large penalty -Z for missing deadline
        alpha_weight = 0.5  # Weight for delay
        beta_weight = 0.5   # Weight for energy
        
        if delay > task.max_delay_d:
            reward = -Z_penalty
        else:
            reward = -(alpha_weight * delay + beta_weight * energy)
            
        # Move to next task
        self.current_task_idx += 1
        done = (self.current_task_idx >= len(self.current_tasks))
        
        # Get next state
        next_state = build_state(self.current_vehicle, self.current_tasks, self.rsus)
        
        info = {
            "delay": delay,
            "energy": energy,
            "missed_deadline": delay > task.max_delay_d
        }
        
        return next_state, float(reward), done, info

    def reset(self) -> np.ndarray:
        """
        Resets the environment.
        In a full implementation, this triggers a step in SUMO and generates new tasks.
        """
        self.current_task_idx = 0
        
        # Return a zeroed state for now to satisfy the gym interface
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def render(self, mode='human'):
        pass
