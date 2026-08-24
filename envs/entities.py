from dataclasses import dataclass, field
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
    trajectory_history: list = field(default_factory=list)  # needed for mobility model inference

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

    def __post_init__(self):
        """Coerce all fields to correct numeric types.

        PyYAML's safe_load can parse scientific-notation values (e.g. 2.0e6)
        as strings in some environments (Python 3.13 on Linux). This guard
        ensures random.uniform() and arithmetic always receive proper floats/ints.
        """
        # Float range lists
        self.vehicle_speed_range      = [float(v) for v in self.vehicle_speed_range]
        self.rsu_cpu_capacity_range   = [float(v) for v in self.rsu_cpu_capacity_range]
        self.task_size_range          = [float(v) for v in self.task_size_range]
        self.task_deadline_range      = [float(v) for v in self.task_deadline_range]
        self.bandwidth_v2r_range      = [float(v) for v in self.bandwidth_v2r_range]
        # Int range lists
        self.num_vehicles_range            = [int(v) for v in self.num_vehicles_range]
        self.num_tasks_per_vehicle_range   = [int(v) for v in self.num_tasks_per_vehicle_range]
        # Scalar floats
        self.num_rsus           = int(self.num_rsus)
        self.rsu_comm_range     = float(self.rsu_comm_range)
        self.bandwidth_r2r      = float(self.bandwidth_r2r)
        self.tx_power_vehicle   = float(self.tx_power_vehicle)
        self.tx_power_rsu       = float(self.tx_power_rsu)
        self.noise_power        = float(self.noise_power)
        self.fixed_loss_k       = float(self.fixed_loss_k)
        self.path_loss_factor   = float(self.path_loss_factor)
        self.alpha              = float(self.alpha)
        self.beta               = float(self.beta)
        self.penalty_z          = float(self.penalty_z)
        self.max_task_cpu       = float(self.max_task_cpu)
        self.epsilon            = float(self.epsilon)