import numpy as np

class GreedyPolicy:
    def __init__(self, config):
        """
        Greedy load-balancing strategy (Section V-B).
        Targets the RSU within range with the shortest current t_wait.
        """
        self.config = config
        self.max_tasks = config.num_tasks_per_vehicle_range[0]
        self.num_rsus = config.num_rsus
        # Calculate the starting index of RSU features in the normalized state vector
        self.rsu_start_idx = 4 + (self.max_tasks * 4)

    def select_action(self, state: np.ndarray, env_info=None) -> int:
        avg_cpu_demand = (1.0e6 + self.config.max_task_cpu * 1.0e6) / 2.0
        
        # Normalization factors used in state_builder.py
        map_length = 2400.0
        max_cpu = 4.0e9
        max_queue_cycles = 100.0 * (self.config.max_task_cpu * 1e6)
        
        v_x = state[0] * map_length
        v_y = state[1] * map_length
        
        min_dist = float('inf')
        target_rsu_idx = 0
        
        min_wait = float('inf')
        best_rsu_idx = 0
        
        for i in range(self.num_rsus):
            idx = self.rsu_start_idx + i * 5
            r_x = state[idx] * map_length
            r_y = state[idx + 1] * map_length
            cpu_cap = state[idx + 2] * max_cpu
            q_cycles = state[idx + 3] * max_queue_cycles
            
            # Determine the primary (nearest) RSU just like the environment does
            dist = np.sqrt((v_x - r_x)**2 + (v_y - r_y)**2)
            if dist < min_dist:
                min_dist = dist
                target_rsu_idx = i
                
            # Estimate t_wait for this RSU
            t_wait = q_cycles / cpu_cap if cpu_cap > 0 else float('inf')
            if t_wait < min_wait:
                min_wait = t_wait
                best_rsu_idx = i
                
        # Decision Logic defined in Phase 2
        # action 0 forces Case 1 (Standalone) at the primary RSU.
        if best_rsu_idx == target_rsu_idx:
            return 0
            
        # action > 0 triggers Case 2 (Collaborative) with RSU action-1
        return best_rsu_idx + 1
