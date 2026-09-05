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

class ScalableList(list):
    """
    A list subclass that supports element-wise division and multiplication by scalars.
    Provides seamless compatibility for notebook cells dividing range lists (e.g. bandwidth/1e6).
    """
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return ScalableList([x / other for x in self])
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return ScalableList([other / x for x in self])
        return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, (int, float)):
            return ScalableList([x // other for x in self])
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, float):
            return ScalableList([x * other for x in self])
        return super().__mul__(other)


class FrequencyVal(float):
    """
    Float frequency in Hz (e.g. 1.0e9) that defaults to GHz representation when
    converted to str directly without a format specifier (e.g. in f'{f_v} GHz').
    """
    def __str__(self):
        if self >= 1e9 and self % 1e9 == 0:
            return f"{self / 1e9:.1f}"
        return super().__str__()

    def __format__(self, format_spec):
        if not format_spec:
            return str(self)
        return super().__format__(format_spec)


@dataclass
class SimulationConfig:
    num_vehicles_range: List[int] = field(default_factory=lambda: [10, 30])
    num_rsus: int = 6
    vehicle_speed_range: List[float] = field(default_factory=lambda: [30.0, 40.0])
    rsu_cpu_capacity_range: List[float] = field(default_factory=lambda: [1.0e9, 4.0e9])
    num_tasks_per_vehicle_range: List[int] = field(default_factory=lambda: [20, 40])
    task_size_range: List[float] = field(default_factory=lambda: [2.0e6, 5.0e6])
    task_deadline_range: List[float] = field(default_factory=lambda: [20.0, 30.0])
    rsu_comm_range: float = 400.0
    tx_power_vehicle: float = 0.01          # 10 dBm
    tx_power_rsu: float = 100.0             # 50 dBm
    bandwidth_v2r_range: List[float] = field(default_factory=lambda: [20.0e6, 100.0e6])
    bandwidth_r2r: float = 50.0e6
    noise_power: float = 0.001
    fixed_loss_k: float = 1000.0
    path_loss_factor: float = 2.0
    max_task_cpu: float = 10.0
    compute_power_rsu: float = 50.0
    alpha: float = 0.3
    beta: float = 0.7
    epsilon: float = 0.5
    penalty_z: float = 100.0
    vehicle_cpu_capacity_val: float = 1.0e9 # Table III: 1.0 GHz default

    def __post_init__(self):
        """
        Defensive type coercion: PyYAML occasionally loads scientific-notation floats
        as strings in some environments (Python 3.13 on Linux). This guard
        ensures random.uniform() and arithmetic always receive proper floats/ints.
        """
        # Float range lists with ScalableList for element-wise operations
        self.vehicle_speed_range      = ScalableList([float(v) for v in self.vehicle_speed_range])
        self.rsu_cpu_capacity_range   = ScalableList([float(v) for v in self.rsu_cpu_capacity_range])
        self.task_size_range          = ScalableList([float(v) for v in self.task_size_range])
        self.task_deadline_range      = ScalableList([float(v) for v in self.task_deadline_range])
        self.bandwidth_v2r_range      = ScalableList([float(v) for v in self.bandwidth_v2r_range])
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
        self.vehicle_cpu_capacity_val = float(self.vehicle_cpu_capacity_val)

    @property
    def vehicle_cpu_capacity(self) -> float:
        """Vehicle CPU capacity in Hz (Table III: 1.0 GHz)."""
        return FrequencyVal(self.vehicle_cpu_capacity_val)

    @property
    def task_max_delay_range(self) -> ScalableList:
        """Alias for task_deadline_range (Table III: [20.0, 30.0] s)."""
        return self.task_deadline_range

    @property
    def tx_power_v2r_w(self) -> float:
        """Alias for tx_power_vehicle in Watts (Table III: 0.01 W / 10 dBm)."""
        return self.tx_power_vehicle

    @property
    def tx_power_r2r_w(self) -> float:
        """Alias for tx_power_rsu in Watts (Table III: 100.0 W / 50 dBm)."""
        return self.tx_power_rsu

    @property
    def bandwidth_v2r_hz(self) -> ScalableList:
        """Alias for bandwidth_v2r_range in Hz (Table III: [20.0e6, 100.0e6] Hz)."""
        return self.bandwidth_v2r_range

    @property
    def bandwidth_r2r_hz(self) -> float:
        """Alias for bandwidth_r2r in Hz (Table III: 50.0e6 Hz)."""
        return self.bandwidth_r2r

    @property
    def noise_power_w(self) -> float:
        """Alias for noise_power in Watts (Table III: 0.001 W)."""
        return self.noise_power