import os
import copy
from typing import Tuple, Optional
import numpy as np
from envs.vec_env import VECEnv, TRAJ_HISTORY_LEN
from envs.entities import Vehicle, Task, SimulationConfig, RSU
from utils.realization import load_realization

class FrozenVECEnv(VECEnv):
    """
    A VECEnv subclass that executes strictly from a frozen realization trace.
    It mathematically guarantees that algorithms (e.g., CoTOP, DDQN) cannot mutate
    exogenous state (task generation, vehicle mobility, geometry, random seeds).
    SUMO and TaskGenerator are bypassed entirely.
    """
    def __init__(
        self,
        config: SimulationConfig,
        realization_path: str,
        use_mobility_model: bool = True,
        use_priority: bool = True,
        priority_mode: str = "paper_literal",
        coverage_mode: str = "completion_position"
    ):
        # We don't want to start SUMO for FrozenVECEnv, so we set port=None
        # But we need to load the realization first
        self.realization = load_realization(realization_path)
        self.realization_trace = self.realization["vehicle_trace"]
        self.realization_tasks = self.realization["task_trace"]
        
        # Override config parameters strictly based on realization
        self.frozen_seed = self.realization["seed"]
        self.frozen_geometry = self.realization["geometry"]
        
        super().__init__(
            config=config,
            port=None, # Disabled SUMO port
            use_mobility_model=use_mobility_model,
            use_priority=use_priority,
            priority_mode=priority_mode,
            coverage_mode=coverage_mode,
            scenario_geometry=self.frozen_geometry,
            spatial_graph_radius=200.0,
            seed=self.frozen_seed,
            max_vehicles=len(self.realization_tasks.keys())
        )
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """
        Bypasses SUMO initialization and uses strictly frozen positions and tasks.
        """
        if seed is not None and seed != self.frozen_seed:
            raise ValueError(f"FrozenVECEnv restricts seed overriding. Bound to frozen seed {self.frozen_seed}")
            
        self.sim_time = 0.0
        self.active_vehicles = {}
        self.vehicle_tasks = {}
        self.generated_vehicle_ids = set()
        self.pending_tasks = []
        self.completed_tasks = []
        self.failed_tasks = []
        self.current_vehicle = None
        self.current_tasks = []
        self.current_task_idx = 0
        
        # RSU geometry is static, initialization logic remains
        from utils.scenario_geometry import get_rsu_positions
        # Pass scenario_mode as geometry, mock conn as None since SUMO is not running
        positions = get_rsu_positions(self.config.num_rsus, None, scenario_mode=self.scenario_geometry)
        self.rsus = [
            RSU(i, positions[i], self.config.rsu_cpu_capacity_range[0], 0.0, self.config.tx_power_rsu)
            for i in range(self.config.num_rsus)
        ]

        # Advance timeline until the first tasks appear
        while len(self.pending_tasks) == 0 and self.sim_time < self.max_sim_time:
            self._advance_sumo_time_slot()

        return self._get_obs(), {}
        
    def _advance_sumo_time_slot(self) -> bool:
        """
        Advances based on the frozen realization trace instead of SUMO.
        """
        trace_key = str(int(self.sim_time))
        if trace_key not in self.realization_trace:
            return False # End of trace
            
        vehicles_data = self.realization_trace[trace_key]
        self.sim_time += 1.0

        # Drain shared RSU queues
        for rsu in self.rsus:
            rsu.queued_cpu_cycles = max(0.0, rsu.queued_cpu_cycles - rsu.cpu_capacity_f * 1.0)
            
        current_step_ids = set(vehicles_data.keys())

        # Clean up departed vehicles
        departed_ids = set(self.active_vehicles.keys()) - current_step_ids
        for dep_id in departed_ids:
            # Unscheduled tasks for departed vehicle transition to FAILED_DEPARTURE
            for task in self.vehicle_tasks.get(dep_id, []):
                if task.priority >= 0:
                    self.failed_tasks.append((task, "FAILED_DEPARTURE"))
            self.active_vehicles.pop(dep_id, None)
            self.vehicle_tasks.pop(dep_id, None)

        # Update or register active vehicles
        for v_id, v_data in vehicles_data.items():
            if v_id not in self.active_vehicles:
                self.generated_vehicle_ids.add(v_id)
                veh = Vehicle(
                    v_id=v_id,
                    pos=tuple(v_data['pos']),
                    speed=v_data['speed'],
                    dwell_time_T_stay=0.0,
                    trajectory_history=[tuple(v_data['pos'])]
                )
                self.active_vehicles[v_id] = veh
                
                # Fetch frozen tasks instead of generating
                task_dicts = self.realization_tasks.get(v_id, [])
                tasks = []
                for td in task_dicts:
                    t = Task(td["task_id"], v_id, td["size_rho"], td["cpu_phi"], td["max_delay_d"])
                    tasks.append(t)
                self.vehicle_tasks[v_id] = tasks
            else:
                veh = self.active_vehicles[v_id]
                veh.pos = tuple(v_data['pos'])
                veh.speed = v_data['speed']
                veh.trajectory_history = (veh.trajectory_history + [tuple(v_data['pos'])])[-TRAJ_HISTORY_LEN:]

        # Update dwell times
        self._estimate_all_dwell_times()
        self._rebuild_pending_tasks()
        
        return len(self.active_vehicles) > 0 or len(self.generated_vehicle_ids) < self.max_vehicles
        
    def close(self):
        pass # No SUMO to close
