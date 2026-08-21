from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class Task:
    task_id: int
    vehicle_id: str
    size_rho: float
    cpu_phi: float
    max_delay_d: float
    priority: float = 0.0

@dataclass
class RSU:
    rsu_id: int
    location: Tuple[float, float]
    cpu_capacity_f: float
    queue_length: int
    transmission_power_P_R: float

@dataclass
class Vehicle:
    v_id: str
    pos: Tuple[float, float]
    speed: float
    dwell_time_T_stay: float
    trajectory_history: Optional[list] = None  # needed for mobility model inference

    def __post_init__(self):
        if self.trajectory_history is None:
            self.trajectory_history = []

@dataclass
class SimulationConfig:
    """Matches every key in configs/simulation.yaml exactly."""
    num_vehicles_range: List[int]
    num_rsus: int
    vehicle_speed_range: List[float]
    rsu_cpu_capacity_range: List[float]
    num_tasks_per_vehicle_range: List[int]
    task_size_range: List[float]
    task_deadline_range: List[float]
    bandwidth_v2r_range: List[float]

    rsu_comm_range: float
    bandwidth_r2r: float
    tx_power_vehicle: float
    tx_power_rsu: float
    noise_power: float
    fixed_loss_k: float
    path_loss_factor: float

    alpha: float
    beta: float
    penalty_z: float
    max_task_cpu: float          # was missing — YAML has it, dataclass didn't
    epsilon: float = 0.5          # Eq. 25 reward trade-off, separate from alpha/beta