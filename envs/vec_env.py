import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import os
import math
from typing import Tuple, Dict, List, Set, Optional

from envs.entities import Vehicle, Task, RSU, SimulationConfig
from envs.state_builder import build_state
from envs.comm_model import compute_v2r_rate, compute_r2r_rate
from envs.comp_model import calculate_case1_standalone, calculate_case2_collaboration
from envs.task_generator import TaskGenerator
from utils.task_priority import prioritize_tasks, prioritize_task_queue
from envs.sumo_manager import SumoManager
from utils.seed import set_seed

def get_euclidean_distance(pos_a: Tuple[float, float], pos_b: Tuple[float, float]) -> float:
    return math.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)

MOBILITY_CHECKPOINT = "results/checkpoints/mobility_model.pth"
TRAJ_HISTORY_LEN = 5

class VECEnv(gym.Env):
    """
    Multi-Vehicle Vehicular Edge Computing Gymnasium Environment for CoTOP.
    
    Supports:
    - Genuine concurrent multi-vehicle simulation with SUMO time advancement.
    - Global task prioritization (Eq. 23) across all active vehicles.
    - Shared RSU computational queues (Eq. 5) experiencing realistic multi-vehicle contention.
    - Full backward-compatible 114-dimensional observation space and 7-dimensional action space.
    """
    def __init__(self, config: SimulationConfig, port: int = None, use_mobility_model: bool = True, use_priority: bool = True, seed: int = None, max_vehicles: Optional[int] = None):
        super(VECEnv, self).__init__()
        self.config = config
        self.port = port
        self.use_mobility_model = use_mobility_model
        self.use_priority = use_priority
        self.seed = seed
        if self.seed is not None:
            set_seed(self.seed)
        self.max_vehicles = max_vehicles if max_vehicles is not None else getattr(self.config, 'max_vehicles', self.config.num_vehicles_range[1])

        self.action_space = spaces.Discrete(self.config.num_rsus + 1)
        n_tasks = self.config.num_tasks_per_vehicle_range[0]
        self.obs_dim = 4 + (n_tasks * 4) + (self.config.num_rsus * 5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.sumo_manager = SumoManager("sumo_config/hangzhou.sumocfg", port=self.port, use_gui=False, seed=seed)
        self.sumo_started = False
        
        # Multi-vehicle state
        self.active_vehicles: Dict[str, Vehicle] = {}
        self.generated_vehicle_ids: Set[str] = set()
        self.pending_tasks: List[Tuple[Vehicle, Task]] = []
        self.rsus: List[RSU] = []
        self.sim_time: float = 0.0
        self.max_sim_time: float = 300.0
        
        # Compatibility references for current task/vehicle
        self.current_vehicle: Optional[Vehicle] = None
        self.current_tasks: List[Task] = []
        self.current_task_idx: int = 0
        
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
                # Normalize trajectory coordinates to [0, 1] for GAT-GRU model
                traj_norm = traj / 2400.0
                x_seq = torch.FloatTensor(traj_norm).unsqueeze(0)
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                with torch.no_grad():
                    predictions = self.mobility_model(x_seq, edge_index)
                future_pos_norm = predictions[0, -1].numpy()
                future_pos = future_pos_norm * 2400.0
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

    def _get_obs(self) -> np.ndarray:
        if len(self.pending_tasks) > 0:
            curr_veh, curr_task = self.pending_tasks[0]
            veh_tasks = [t for v, t in self.pending_tasks if v.v_id == curr_veh.v_id]
            return build_state(curr_veh, veh_tasks, self.rsus, self.config)
        elif self.current_vehicle is not None:
            return build_state(self.current_vehicle, self.current_tasks, self.rsus, self.config)
        return np.zeros(self.obs_dim, dtype=np.float32)

    def _advance_sumo_time_slot(self) -> bool:
        """
        Advances SUMO by 1.0 second (one time slot), drains RSU queues according
        to their service capacity (F_m * dt), updates active vehicle positions,
        computes dwell times, generates tasks for new vehicle arrivals, and
        re-prioritizes the global pending task queue.
        """
        self.sumo_manager.step()
        self.sim_time += 1.0

        # Drain shared RSU queues for elapsed time Delta t = 1.0 s
        dt = 1.0
        for rsu in self.rsus:
            rsu.queued_cpu_cycles = max(0.0, rsu.queued_cpu_cycles - rsu.cpu_capacity_f * dt)

        # Retrieve active vehicles from SUMO
        vehicles_data = self.sumo_manager.get_vehicle_data()
        active_ids = set(vehicles_data.keys())

        # Update or register active vehicles
        for v_id, v_data in vehicles_data.items():
            if v_id in self.active_vehicles:
                veh = self.active_vehicles[v_id]
                veh.pos = v_data['pos']
                veh.speed = v_data['speed']
                veh.trajectory_history = (veh.trajectory_history + [v_data['pos']])[-TRAJ_HISTORY_LEN:]
            else:
                veh = Vehicle(
                    v_id=v_id,
                    pos=v_data['pos'],
                    speed=v_data['speed'],
                    dwell_time_T_stay=0.0,
                    trajectory_history=[v_data['pos']],
                )
                self.active_vehicles[v_id] = veh

                # Generate tasks for newly active vehicle up to max_vehicles limit
                if len(self.generated_vehicle_ids) < self.max_vehicles and v_id not in self.generated_vehicle_ids:
                    new_tasks = self.task_gen.generate_tasks_for_vehicle(v_id)
                    for t in new_tasks:
                        self.pending_tasks.append((veh, t))
                    self.generated_vehicle_ids.add(v_id)

            # Update dwell time relative to primary RSU
            nearest_rsu = min(self.rsus, key=lambda r: get_euclidean_distance(veh.pos, r.location))
            veh.dwell_time_T_stay = self._estimate_dwell_time(veh, nearest_rsu)

        # Remove departed vehicles from active_vehicles
        departed_ids = set(self.active_vehicles.keys()) - active_ids
        for d_id in departed_ids:
            del self.active_vehicles[d_id]

        # Prioritize all pending tasks globally across all active vehicles (Eq. 23)
        if self.use_priority and len(self.pending_tasks) > 0:
            self.pending_tasks = prioritize_task_queue(
                self.pending_tasks,
                alpha=self.config.alpha,
                beta=self.config.beta
            )

        if len(self.pending_tasks) > 0:
            self.current_vehicle = self.pending_tasks[0][0]
            self.current_tasks = [t for v, t in self.pending_tasks if v.v_id == self.current_vehicle.v_id]
            self.current_task_idx = 0

        has_future_vehicles = (len(self.generated_vehicle_ids) < self.max_vehicles)
        return (len(self.active_vehicles) > 0 or len(self.pending_tasks) > 0 or has_future_vehicles)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if len(self.pending_tasks) == 0:
            while len(self.pending_tasks) == 0 and self.sim_time < self.max_sim_time:
                has_more = self._advance_sumo_time_slot()
                if not has_more:
                    break
            if len(self.pending_tasks) == 0:
                return self._get_obs(), 0.0, True, False, {}

        # Pop the highest-priority pending task
        current_vehicle, task = self.pending_tasks.pop(0)
        self.current_vehicle = current_vehicle
        self.current_tasks = [t for v, t in self.pending_tasks if v.v_id == current_vehicle.v_id]
        self.current_task_idx = 0

        # Primary RSU is the nearest RSU
        target_rsu = min(self.rsus, key=lambda r: get_euclidean_distance(current_vehicle.pos, r.location))

        v2r_distance = get_euclidean_distance(current_vehicle.pos, target_rsu.location)
        w_v2r = compute_v2r_rate(
            distance=v2r_distance,
            bandwidth_B=self.config.bandwidth_v2r_range[0],
            power_P_V=self.config.tx_power_vehicle,
            noise_power=self.config.noise_power,
            fixed_loss_k=self.config.fixed_loss_k,
            path_loss_factor=self.config.path_loss_factor
        )

        dwell_time = current_vehicle.dwell_time_T_stay

        # Eq 5: t_wait = N^{queue} / F^{RSU}
        t_wait_target = target_rsu.queued_cpu_cycles / target_rsu.cpu_capacity_f if target_rsu.cpu_capacity_f > 0 else 0.0

        # Action Decision Logic
        if action == 0:
            # Force Case 1 (Standalone)
            standalone_delay, energy = calculate_case1_standalone(
                task_size_rho=task.size_rho,
                task_cpu_phi=task.cpu_phi,
                w_v2r=w_v2r,
                rsu_cpu_f=target_rsu.cpu_capacity_f,
                power_v=self.config.tx_power_vehicle,
                compute_power_rsu=self.config.compute_power_rsu,
                t_wait=t_wait_target
            )
            target_rsu.queued_cpu_cycles += task.cpu_phi
            case_used = 1
            comm_delay = (task.size_rho * 8.0) / w_v2r
            comp_delay = task.cpu_phi / target_rsu.cpu_capacity_f
            wait_delay = t_wait_target
        else:
            # action > 0 triggers Case 2 (Collaborative) with RSU action-1
            secondary_rsu = self.rsus[action - 1]
            if secondary_rsu.rsu_id == target_rsu.rsu_id:
                # If agent selects primary RSU for collaboration, fall back to standalone
                standalone_delay, energy = calculate_case1_standalone(
                    task_size_rho=task.size_rho,
                    task_cpu_phi=task.cpu_phi,
                    w_v2r=w_v2r,
                    rsu_cpu_f=target_rsu.cpu_capacity_f,
                    power_v=self.config.tx_power_vehicle,
                    compute_power_rsu=self.config.compute_power_rsu,
                    t_wait=t_wait_target
                )
                target_rsu.queued_cpu_cycles += task.cpu_phi
                case_used = 1
                comm_delay = (task.size_rho * 8.0) / w_v2r
                comp_delay = task.cpu_phi / target_rsu.cpu_capacity_f
                wait_delay = t_wait_target
            else:
                r2r_distance = get_euclidean_distance(target_rsu.location, secondary_rsu.location)
                w_r2r = compute_r2r_rate(
                    distance=r2r_distance,
                    bandwidth_B=self.config.bandwidth_r2r,
                    power_P_R=self.config.tx_power_rsu,
                    noise_power=self.config.noise_power,
                    fixed_loss_k=self.config.fixed_loss_k,
                    path_loss_factor=self.config.path_loss_factor
                )

                # Eq 10: Secondary RSU queue wait time
                t_wait_secondary = secondary_rsu.queued_cpu_cycles / secondary_rsu.cpu_capacity_f if secondary_rsu.cpu_capacity_f > 0 else 0.0

                standalone_delay, energy = calculate_case2_collaboration(
                    task_size_rho=task.size_rho,
                    task_cpu_phi=task.cpu_phi,
                    w_v2r=w_v2r,
                    w_r2r=w_r2r,
                    rsu1_cpu_f=target_rsu.cpu_capacity_f,
                    rsu2_cpu_f=secondary_rsu.cpu_capacity_f,
                    t1_dwell_time=dwell_time,
                    power_v=self.config.tx_power_vehicle,
                    tx_power_rsu1=self.config.tx_power_rsu,
                    compute_power_rsu1=self.config.compute_power_rsu,
                    compute_power_rsu2=self.config.compute_power_rsu,
                    t_wait=t_wait_secondary
                )
                
                # Proportional queue allocation
                t_comp1 = task.cpu_phi / target_rsu.cpu_capacity_f if target_rsu.cpu_capacity_f > 0 else 0.0
                part_ratio = target_rsu.cpu_capacity_f / (target_rsu.cpu_capacity_f + secondary_rsu.cpu_capacity_f) if (target_rsu.cpu_capacity_f + secondary_rsu.cpu_capacity_f) > 0 else 0.5
                t1 = min(dwell_time, part_ratio * t_comp1) if dwell_time >= t_comp1 else max(dwell_time, 0.0)
                cpu1 = min(target_rsu.cpu_capacity_f * t1, task.cpu_phi)
                cpu2 = max(task.cpu_phi - cpu1, 0.0)
                
                target_rsu.queued_cpu_cycles += cpu1
                secondary_rsu.queued_cpu_cycles += cpu2
                case_used = 2
                
                t2_inter = (task.size_rho * (cpu2 / task.cpu_phi) * 8.0) / w_r2r if task.cpu_phi > 0 else 0.0
                comm_delay = ((task.size_rho * 8.0) / w_v2r) + t2_inter
                comp_delay = max(t1, cpu2 / secondary_rsu.cpu_capacity_f if secondary_rsu.cpu_capacity_f > 0 else 0.0)
                wait_delay = t_wait_secondary

        eps = getattr(self.config, 'epsilon', 0.5)
        if standalone_delay > task.max_delay_d:
            reward = -self.config.penalty_z
        else:
            reward = -(eps * standalone_delay + (1.0 - eps) * energy)

        info = {
            "delay": standalone_delay,
            "energy": energy,
            "case": case_used,
            "comm_delay": comm_delay,
            "comp_delay": comp_delay,
            "wait_delay": wait_delay,
            "v_id": current_vehicle.v_id,
            "task_id": task.task_id,
            "completed": bool(standalone_delay <= task.max_delay_d),
            "rsu_queues": [float(r.queued_cpu_cycles) for r in self.rsus],
            "active_vehicles_count": len(self.active_vehicles),
            "pending_tasks_count": len(self.pending_tasks),
        }

        # If no more pending tasks in current batch, advance SUMO
        if len(self.pending_tasks) == 0:
            while len(self.pending_tasks) == 0 and self.sim_time < self.max_sim_time:
                has_more = self._advance_sumo_time_slot()
                if not has_more:
                    break

        terminated = (len(self.pending_tasks) == 0)
        return self._get_obs(), float(reward), terminated, False, info

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if seed is not None:
            self.seed = seed
            set_seed(seed)
        elif self.seed is not None:
            set_seed(self.seed)

        if self.task_gen:
            self.task_gen._task_counter = 0

        if options and 'max_vehicles' in options:
            self.max_vehicles = int(options['max_vehicles'])
        else:
            self.max_vehicles = getattr(self.config, 'max_vehicles', self.config.num_vehicles_range[1])

        if not self.sumo_started:
            self.sumo_manager.start_simulation()
            self.sumo_started = True
        else:
            self.sumo_manager.reload_simulation()

        from utils.scenario_geometry import get_rsu_positions
        positions = get_rsu_positions(self.config.num_rsus, getattr(self.sumo_manager, 'conn', None))
        self.rsus = [
            RSU(i, positions[i], self.config.rsu_cpu_capacity_range[0], 0.0, self.config.tx_power_rsu)
            for i in range(self.config.num_rsus)
        ]

        self.active_vehicles = {}
        self.generated_vehicle_ids = set()
        self.pending_tasks = []
        self.sim_time = 0.0
        self.current_vehicle = None
        self.current_tasks = []
        self.current_task_idx = 0

        # Advance SUMO until active vehicles enter and generate pending tasks
        while len(self.pending_tasks) == 0 and self.sim_time < self.max_sim_time:
            self._advance_sumo_time_slot()

        return self._get_obs(), {}

    def close(self):
        if self.sumo_manager:
            self.sumo_manager.close()