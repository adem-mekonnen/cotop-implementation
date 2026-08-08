from dataclasses import dataclass
from typing import Tuple

@dataclass
class Task:
    task_id: int
    vehicle_id: int
    size_rho: float  # Eq 3
    cpu_phi: float   # Eq 4
    max_delay_d: float # Eq 14b
    priority: int

@dataclass
class RSU:
    rsu_id: int
    location: Tuple[float, float] # (x, y)
    cpu_capacity_f: float         # Table III
    queue_length: int             # N_queue
    transmission_power_P_R: float

@dataclass
class Vehicle:
    v_id: int
    pos: Tuple[float, float]      # (x, y)
    speed: float
    dwell_time_T_stay: float      # Eq 23

@dataclass
class SimulationConfig:
    # Constants from Table III
    bandwidth_B: float
    noise_power_sigma2: float
    # Note: Add other constants from Table III here as needed
