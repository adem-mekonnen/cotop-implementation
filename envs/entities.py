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
    queued_cpu_cycles: float
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
    """Matches every key in configs/paper_parameters.yaml exactly with Table III defaults."""
    num_vehicles_range: List[int] = field(default_factory=lambda: [10, 30])
    num_rsus: int = 6
    vehicle_speed_range: List[float] = field(default_factory=lambda: [30.0, 40.0])
    rsu_cpu_capacity_range: List[float] = field(default_factory=lambda: [1.0e9, 4.0e9])
    num_tasks_per_vehicle_range: List[int] = field(default_factory=lambda: [20, 40])
    task_size_range: List[float] = field(default_factory=lambda: [2.0e6, 5.0e6])
    task_deadline_range: List[float] = field(default_factory=lambda: [20.0, 30.0])
    bandwidth_v2r_range: List[float] = field(default_factory=lambda: [20.0e6, 100.0e6])

    rsu_comm_range: float = 400.0
    bandwidth_r2r: float = 50.0e6
    tx_power_vehicle: float = 0.01
    tx_power_rsu: float = 100.0
    compute_power_rsu: float = 50.0
    noise_power: float = 0.001
    fixed_loss_k: float = 1000.0
    path_loss_factor: float = 2.0

    alpha: float = 0.3
    beta: float = 0.7
    penalty_z: float = 100.0
    max_task_cpu: float = 10.0
    epsilon: float = 0.5

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
        self.compute_power_rsu  = float(self.compute_power_rsu)
        self.noise_power        = float(self.noise_power)
        self.fixed_loss_k       = float(self.fixed_loss_k)
        self.path_loss_factor   = float(self.path_loss_factor)
        self.alpha              = float(self.alpha)
        self.beta               = float(self.beta)
        self.penalty_z          = float(self.penalty_z)
        self.max_task_cpu       = float(self.max_task_cpu)
        self.epsilon            = float(self.epsilon)