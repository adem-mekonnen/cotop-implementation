from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class Task:
    task_id: int
    vehicle_id: str
    size_rho: float       # Eq 3: bytes
    cpu_phi: float        # Eq 4: cycles
    max_delay_d: float    # Eq 14b: seconds
    priority: float = 0.0 # Eq 23 (Result of prioritization)

@dataclass
class RSU:
    rsu_id: int
    location: Tuple[float, float] # (x, y) coordinates
    cpu_capacity_f: float         # F^{RSU} in cycles/s
    queue_length: int             # N^{queue}
    transmission_power_P_R: float # P^R in Watts

@dataclass
class Vehicle:
    v_id: str
    pos: Tuple[float, float]      # (x, y) coordinates
    speed: float                  # v in m/s
    dwell_time_T_stay: float      # T^{stay} in seconds (Eq 23)

@dataclass
class SimulationConfig:
    """Matches every key in configs/simulation.yaml exactly."""
    # Table III Ranges
    num_vehicles_range: List[int]
    num_rsus: int
    vehicle_speed_range: List[float]
    rsu_cpu_capacity_range: List[float]
    num_tasks_per_vehicle_range: List[int]
    task_size_range: List[float]
    task_deadline_range: List[float]
    bandwidth_v2r_range: List[float]
    
    # Table III Constants
    rsu_comm_range: float
    bandwidth_r2r: float
    tx_power_vehicle: float
    tx_power_rsu: float
    noise_power: float
    fixed_loss_k: float
    path_loss_factor: float
    
    # Weights and Penalties
    alpha: float
    beta: float
    penalty_z: float