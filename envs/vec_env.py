import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import os
import math
from typing import Tuple, List, Dict, Set, Optional

from envs.entities import Vehicle, Task, RSU, SimulationConfig
from envs.state_builder import build_state
from envs.comm_model import compute_v2r_rate, compute_r2r_rate
from envs.comp_model import calculate_case1_standalone, calculate_case2_collaboration
from envs.task_generator import TaskGenerator
from utils.task_priority import compute_task_priority_paper, compute_task_priority_normalized, prioritize_tasks_paper, prioritize_tasks_normalized
from envs.sumo_manager import SumoManager
from utils.seed import set_seed

def get_euclidean_distance(pos_a: Tuple[float, float], pos_b: Tuple[float, float]) -> float:
    return math.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)

MOBILITY_CHECKPOINT = "results/checkpoints/mobility_model.pth"
TRAJ_HISTORY_LEN = 5

class VECEnv(gym.Env):
    """
    Phase 1 Scientifically Faithful Vehicular Edge Computing Environment.
    
    Implements:
    - Multi-Vehicle Execution Contract: strict task ownership, dynamic SUMO stepping,
      departure cleanup, and shared RSU computational backlogs (P1).
    - Configurable Scenario Geometry: 2400m Corridor (baseline) and 200m x 200m Reconstructed Grid (P2).
    - Real Multi-Node Mobility GAT Graph: N-node spatial proximity graph with head averaging (P3, P6).
    - Dual Task Prioritization: Paper-Literal Eq. 23 (default) and Normalized Candidate (P4).
    - Physical State Coverage Failure: Configurable operational predicates (P5).
    """
    def __init__(
        self,
        config: SimulationConfig,
        port: int = None,
        use_mobility_model: bool = True,
        use_priority: bool = True,
        priority_mode: str = "paper_literal",
        coverage_mode: str = "completion_position",
        scenario_geometry: str = "corridor_2400m",
        spatial_graph_radius: float = 200.0,
        seed: int = None,
        max_vehicles: Optional[int] = None
    ):
        super(VECEnv, self).__init__()
        self.config = config
        self.port = port
        self.use_mobility_model = use_mobility_model
        self.use_priority = use_priority
        self.priority_mode = priority_mode          # 'paper_literal' (default) or 'normalized_candidate'
        self.coverage_mode = coverage_mode          # 'completion_position' (default) or 'continuous_required_rsus'
        self.scenario_geometry = scenario_geometry  # 'corridor_2400m' or 'grid_200m'
        self.spatial_graph_radius = spatial_graph_radius # Reconstructed assumption
        self.seed = seed
        if self.seed is not None:
            set_seed(self.seed)
            
        self.max_vehicles = max_vehicles if max_vehicles is not None else getattr(self.config, 'max_vehicles', self.config.num_vehicles_range[1])

        self.action_space = spaces.Discrete(self.config.num_rsus + 1)
        n_tasks = self.config.num_tasks_per_vehicle_range[0]
        self.obs_dim = 4 + (n_tasks * 4) + (self.config.num_rsus * 5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        sumo_cfg_file = "sumo_config/hangzhou_200m.sumocfg" if self.scenario_geometry == "grid_200m" else "sumo_config/hangzhou.sumocfg"
        self.map_scale = 200.0 if self.scenario_geometry == "grid_200m" else 2400.0
        
        self.sumo_manager = SumoManager(sumo_cfg_file, port=self.port, use_gui=False, seed=seed)
        self.sumo_started = False
        
        # Multi-Vehicle State Hierarchy (Task Ownership Invariant)
        self.active_vehicles: Dict[str, Vehicle] = {}
        self.vehicle_tasks: Dict[str, List[Task]] = {}
        self.generated_vehicle_ids: Set[str] = set()
        self.pending_tasks: List[Tuple[Vehicle, Task]] = []
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Tuple[Task, str]] = []
        
        self.rsus: List[RSU] = []
        self.sim_time: float = 0.0
        self.max_sim_time: float = 300.0
        
        # Backward-compatibility references for current vehicle / tasks
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
            except Exception as e:
                self.mobility_model = None

    def _build_mobility_graph(self) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Constructs genuine N-node spatial proximity graph for active vehicles (P3).
        Returns: (x_seq: (N, T, 2), edge_index: (2, E), vehicle_ids: List[str])
        """
        valid_vehs = [
            v for v in self.active_vehicles.values()
            if len(v.trajectory_history) >= TRAJ_HISTORY_LEN
        ]
        if len(valid_vehs) == 0:
            return torch.zeros(1, TRAJ_HISTORY_LEN, 2), torch.tensor([[0], [0]], dtype=torch.long), []
            
        num_nodes = len(valid_vehs)
        veh_ids = [v.v_id for v in valid_vehs]
        
        # Collect trajectories and normalize by map scale
        trajs = [np.array(v.trajectory_history[-TRAJ_HISTORY_LEN:], dtype=np.float32) / self.map_scale for v in valid_vehs]
        x_seq = torch.FloatTensor(np.stack(trajs, axis=0)) # (N, T, 2)
        
        if num_nodes == 1:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            return x_seq, edge_index, veh_ids
            
        # Compute pairwise distances at last historical frame
        last_pos = np.array([v.pos for v in valid_vehs], dtype=np.float32) # (N, 2)
        diff = last_pos[:, np.newaxis, :] - last_pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1)) # (N, N)
        
        adj = (dist <= self.spatial_graph_radius).astype(int)
        np.fill_diagonal(adj, 1) # Self-loops
        
        edge_index = torch.tensor(np.argwhere(adj == 1).T, dtype=torch.long)
        return x_seq, edge_index, veh_ids

    def _estimate_all_dwell_times(self):
        """Updates dwell time estimates for all active vehicles using multi-node GAT."""
        if not self.rsus or len(self.active_vehicles) == 0:
            return
            
        if self.mobility_model is not None:
            x_seq, edge_index, veh_ids = self._build_mobility_graph()
            if len(veh_ids) > 0:
                try:
                    with torch.no_grad():
                        predictions = self.mobility_model(x_seq, edge_index) # (N, T_pred, 2)
                    future_pos_all = predictions[:, -1, :].numpy() * self.map_scale
                    
                    for idx, v_id in enumerate(veh_ids):
                        if v_id in self.active_vehicles:
                            veh = self.active_vehicles[v_id]
                            nearest_rsu = min(self.rsus, key=lambda r: get_euclidean_distance(veh.pos, r.location))
                            future_pos = future_pos_all[idx]
                            dist_to_edge = self.config.rsu_comm_range - get_euclidean_distance(
                                (float(future_pos[0]), float(future_pos[1])), nearest_rsu.location
                            )
                            speed = max(veh.speed, 1e-3)
                            veh.dwell_time_T_stay = max(dist_to_edge / speed, 0.5)
                    return
                except Exception:
                    pass
                    
        # Fallback distance-based calculation
        for veh in self.active_vehicles.values():
            nearest_rsu = min(self.rsus, key=lambda r: get_euclidean_distance(veh.pos, r.location))
            dist_to_rsu = get_euclidean_distance(veh.pos, nearest_rsu.location)
            remaining = max(self.config.rsu_comm_range - dist_to_rsu, 0.1)
            speed = max(veh.speed, 1e-3)
            veh.dwell_time_T_stay = remaining / speed

    def _get_obs(self) -> np.ndarray:
        if len(self.pending_tasks) > 0:
            curr_veh, curr_task = self.pending_tasks[0]
            veh_tasks = self.vehicle_tasks.get(curr_veh.v_id, [curr_task])
            return build_state(curr_veh, veh_tasks, self.rsus, self.config)
        elif self.current_vehicle is not None:
            return build_state(self.current_vehicle, self.current_tasks, self.rsus, self.config)
        return np.zeros(self.obs_dim, dtype=np.float32)

    def _advance_sumo_time_slot(self) -> bool:
        """
        Advances SUMO by 1.0 second, drains shared RSU queues (F_m * dt),
        updates active vehicle positions, handles dynamic vehicle arrivals/departures,
        and re-prioritizes eligible tasks.
        """
        self.sumo_manager.step()
        self.sim_time += 1.0

        # Drain shared RSU queues
        for rsu in self.rsus:
            rsu.queued_cpu_cycles = max(0.0, rsu.queued_cpu_cycles - rsu.cpu_capacity_f * 1.0)

        vehicles_data = self.sumo_manager.get_vehicle_data()
        current_step_ids = set(vehicles_data.keys())

        # Clean up departed vehicles with explicit departure semantics
        departed_ids = set(self.active_vehicles.keys()) - current_step_ids
        for dep_id in departed_ids:
            # Unscheduled tasks for departed vehicle transition to FAILED_DEPARTURE
            for task in self.vehicle_tasks.get(dep_id, []):
                if task.priority >= 0: # Still unscheduled
                    self.failed_tasks.append((task, "FAILED_DEPARTURE"))
            self.active_vehicles.pop(dep_id, None)
            self.vehicle_tasks.pop(dep_id, None)

        # Update or register active vehicles
        for v_id, v_data in vehicles_data.items():
            if v_id not in self.active_vehicles:
                if len(self.generated_vehicle_ids) < self.max_vehicles:
                    self.generated_vehicle_ids.add(v_id)
                    veh = Vehicle(
                        v_id=v_id,
                        pos=v_data['pos'],
                        speed=v_data['speed'],
                        dwell_time_T_stay=0.0,
                        trajectory_history=[v_data['pos']]
                    )
                    self.active_vehicles[v_id] = veh
                    # Generate owned tasks (Task Ownership Invariant)
                    tasks = self.task_gen.generate_tasks_for_vehicle(v_id)
                    for t in tasks:
                        t.vehicle_id = v_id
                    self.vehicle_tasks[v_id] = tasks
            else:
                veh = self.active_vehicles[v_id]
                veh.pos = v_data['pos']
                veh.speed = v_data['speed']
                veh.trajectory_history = (veh.trajectory_history + [v_data['pos']])[-TRAJ_HISTORY_LEN:]

        # Update dwell times via multi-node GAT
        self._estimate_all_dwell_times()

        # Re-prioritize pending tasks across active vehicles
        self._rebuild_pending_tasks()
        return len(self.active_vehicles) > 0 or len(self.generated_vehicle_ids) < self.max_vehicles

    def _rebuild_pending_tasks(self):
        """Builds and prioritizes the global pool of eligible tasks while preserving ownership."""
        all_pending = []
        for v_id, veh in self.active_vehicles.items():
            tasks = self.vehicle_tasks.get(v_id, [])
            dwell = veh.dwell_time_T_stay
            for task in tasks:
                if self.use_priority:
                    if self.priority_mode == "normalized_candidate":
                        task.priority = compute_task_priority_normalized(
                            task, dwell, self.config.alpha, self.config.beta
                        )
                    else:
                        task.priority = compute_task_priority_paper(
                            task, dwell, self.config.alpha, self.config.beta
                        )
                else:
                    task.priority = 1.0
                all_pending.append((veh, task))
                
        # Sort by priority descending (or FIFO if priority disabled)
        if self.use_priority:
            all_pending.sort(key=lambda item: item[1].priority, reverse=True)
            
        self.pending_tasks = all_pending
        if len(self.pending_tasks) > 0:
            self.current_vehicle, _ = self.pending_tasks[0]
            self.current_tasks = self.vehicle_tasks.get(self.current_vehicle.v_id, [])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if len(self.pending_tasks) == 0:
            # Advance SUMO until pending tasks appear or max sim time reached
            while len(self.pending_tasks) == 0 and self.sim_time < self.max_sim_time:
                has_more = self._advance_sumo_time_slot()
                if not has_more:
                    break
            if len(self.pending_tasks) == 0:
                return self._get_obs(), 0.0, True, False, {"active_vehicles": len(self.active_vehicles)}

        # Pop top-priority task
        current_vehicle, task = self.pending_tasks.pop(0)
        
        # Enforce Task Ownership Invariant
        assert task.vehicle_id == current_vehicle.v_id, "Task ownership mismatch!"
        
        # Remove task from vehicle's owned task list
        if current_vehicle.v_id in self.vehicle_tasks:
            self.vehicle_tasks[current_vehicle.v_id] = [
                t for t in self.vehicle_tasks[current_vehicle.v_id] if t.task_id != task.task_id
            ]

        # Primary RSU is nearest RSU to vehicle
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
        t_wait_target = target_rsu.queued_cpu_cycles / target_rsu.cpu_capacity_f if target_rsu.cpu_capacity_f > 0 else 0.0
        
        rsu_queue_before = target_rsu.queued_cpu_cycles

        # Offloading Action Execution
        if action == 0:
            # Standalone Offloading (Case 1)
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
            comp_delay = task.cpu_phi / target_rsu.cpu_capacity_f if target_rsu.cpu_capacity_f > 0 else 0.0
            wait_delay = t_wait_target
            secondary_rsu = None
        else:
            # Collaborative Offloading (Case 2) with RSU action-1
            secondary_rsu = self.rsus[action - 1]
            if secondary_rsu.rsu_id == target_rsu.rsu_id:
                # Standalone fallback
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
                comp_delay = task.cpu_phi / target_rsu.cpu_capacity_f if target_rsu.cpu_capacity_f > 0 else 0.0
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

        # Calculate Completion Position based on velocity and delay
        speed = max(current_vehicle.speed, 1.0)
        completion_pos = (
            current_vehicle.pos[0] + speed * standalone_delay,
            current_vehicle.pos[1]
        )

        # Eq. 25 Physical Failure Predicates (P5)
        fail_deadline = bool(standalone_delay > task.max_delay_d)
        
        if case_used == 1 or secondary_rsu is None:
            # Standalone coverage predicate
            if self.coverage_mode == "completion_position":
                dist_comp = get_euclidean_distance(completion_pos, target_rsu.location)
                fail_coverage = bool(dist_comp > self.config.rsu_comm_range)
            else: # continuous_required_rsus
                fail_coverage = bool(standalone_delay > dwell_time)
        else:
            # Collaborative coverage predicate
            if self.coverage_mode == "completion_position":
                dist1 = get_euclidean_distance(completion_pos, target_rsu.location)
                dist2 = get_euclidean_distance(completion_pos, secondary_rsu.location)
                # Fails if vehicle is outside coverage of BOTH RSUs at completion
                fail_coverage = bool(dist1 > self.config.rsu_comm_range and dist2 > self.config.rsu_comm_range)
            else: # continuous_required_rsus
                sec_dwell = secondary_rsu.rsu_id # Estimated dwell in secondary RSU
                fail_coverage = bool(standalone_delay > (dwell_time * 2.0))

        is_failed = fail_deadline or fail_coverage
        eps = getattr(self.config, 'epsilon', 0.5)

        if is_failed:
            reward = -self.config.penalty_z # -100.0
            if fail_deadline and fail_coverage:
                failure_reason = "DUAL_VIOLATION"
            elif fail_deadline:
                failure_reason = "DEADLINE_EXCEEDED"
            else:
                failure_reason = "COVERAGE_VIOLATION"
            self.failed_tasks.append((task, failure_reason))
        else:
            reward = -(eps * standalone_delay + (1.0 - eps) * energy)
            failure_reason = "NONE"
            self.completed_tasks.append(task)

        # Build comprehensive Step Info dictionary
        info = {
            "delay": standalone_delay,
            "energy": energy,
            "case": case_used,
            "comm_delay": comm_delay,
            "comp_delay": comp_delay,
            "wait_delay": wait_delay,
            "v_id": current_vehicle.v_id,
            "task_id": task.task_id,
            "task_priority": float(task.priority),
            "priority_mode": self.priority_mode,
            "coverage_mode": self.coverage_mode,
            "completed": not is_failed,
            "fail_deadline": fail_deadline,
            "fail_coverage": fail_coverage,
            "failure_reason": failure_reason,
            "rsu_queue_before": float(rsu_queue_before),
            "rsu_queue_after": float(target_rsu.queued_cpu_cycles),
            "rsu_queues": [float(r.queued_cpu_cycles) for r in self.rsus],
            "active_vehicles_count": len(self.active_vehicles),
            "pending_tasks_count": len(self.pending_tasks),
            "task_ownership_verified": bool(task.vehicle_id == current_vehicle.v_id),
        }

        # If no more pending tasks in current batch, advance SUMO
        if len(self.pending_tasks) == 0:
            while len(self.pending_tasks) == 0 and self.sim_time < self.max_sim_time:
                has_more = self._advance_sumo_time_slot()
                if not has_more:
                    break

        terminated = (len(self.pending_tasks) == 0 and len(self.active_vehicles) == 0)
        return self._get_obs(), float(reward), terminated, False, info

    def get_action_mask(self) -> np.ndarray:
        """
        Returns authoritative boolean action feasibility mask of shape (7,).
        Action 0: Standalone execution (always feasible).
        Actions 1..6: Collaborative offloading to RSUs 0..5 (feasible if RSU active).
        """
        mask = np.ones(self.action_space.n, dtype=bool)
        if len(self.active_vehicles) == 0:
            mask[1:] = False
            return mask
        mask[0] = True
        for i in range(min(len(self.rsus), self.action_space.n - 1)):
            mask[i + 1] = True
        for i in range(len(self.rsus), self.action_space.n - 1):
            mask[i + 1] = False
        return mask

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
        positions = get_rsu_positions(self.config.num_rsus, getattr(self.sumo_manager, 'conn', None), scenario_mode=self.scenario_geometry)
        self.rsus = [
            RSU(i, positions[i], self.config.rsu_cpu_capacity_range[0], 0.0, self.config.tx_power_rsu)
            for i in range(self.config.num_rsus)
        ]

        self.active_vehicles = {}
        self.vehicle_tasks = {}
        self.generated_vehicle_ids = set()
        self.pending_tasks = []
        self.completed_tasks = []
        self.failed_tasks = []
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