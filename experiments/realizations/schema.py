"""
experiments/realizations/schema.py

Authoritative Schema for Controlled Experiment Realizations (Stage 7).
Persists all 9 required exogenous dimensions with cryptographic SHA-256 hash integrity.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any, Optional
import hashlib
import json
import numpy as np


class RealizationEncoder(json.JSONEncoder):
    """JSON Encoder handling NumPy types and float stabilization for deterministic hashing."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(np.round(obj, 8))
        elif isinstance(obj, (float, np.float64)):
            return float(np.round(obj, 8))
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(RealizationEncoder, self).default(obj)


@dataclass
class TaskRealization:
    """Represents a single generated task."""
    task_id: int
    vehicle_id: str
    generation_timestamp: float
    size_rho: float            # Bytes (2.0MB to 5.0MB)
    cpu_phi: float             # Cycles (1.0MCycles to 10.0MCycles)
    max_delay_d: float         # Seconds (20.0s to 30.0s)
    priority_weight: float     # Priority ranking weight [0, 1]


@dataclass
class TrajectoryPoint:
    """A spatial-temporal waypoint along a vehicle's route."""
    timestamp: float
    x: float
    y: float
    speed: float


@dataclass
class VehicleTrajectoryRealization:
    """Represents the complete realized trajectory of a vehicle."""
    vehicle_id: str
    entry_time: float
    initial_position: Tuple[float, float]
    initial_speed: float
    trajectory_points: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class MobilityStateRealization:
    """Represents the spatiotemporal mobility predictions and graph state."""
    vehicle_id: str
    predicted_dwell_time_per_rsu: Dict[str, float] = field(default_factory=dict)
    spatial_proximity_neighbors: List[str] = field(default_factory=list)


@dataclass
class RSUConfigRealization:
    """Represents an RSU's physical infrastructure and configuration."""
    rsu_id: int
    location: Tuple[float, float]
    cpu_capacity_f: float
    initial_queued_cycles: float
    transmission_power_P_R: float
    comm_range: float
    bandwidth_v2r: float
    bandwidth_r2r: float


@dataclass
class InitialConditionsRealization:
    """Represents the system initial state before the first task is scheduled."""
    start_sim_time: float
    num_vehicles: int
    active_vehicle_ids: List[str]
    initial_rsu_backlog_cycles: Dict[str, float]


@dataclass
class WorkloadConfigRealization:
    """Represents the workload parameters."""
    tasks_per_vehicle: int
    total_tasks: int
    task_size_range: List[float]
    task_deadline_range: List[float]
    max_task_cpu: float


@dataclass
class EnvConfigRealization:
    """Represents the environment physical parameters and checksums."""
    env_fingerprint: str
    comm_model_sha256: str
    comp_model_sha256: str
    tx_power_vehicle: float
    tx_power_rsu: float
    noise_power: float
    fixed_loss_k: float
    path_loss_factor: float
    alpha: float
    beta: float
    penalty_z: float
    epsilon: float


@dataclass
class ExperimentRealization:
    """
    Authoritative Container for a complete, deterministic, cryptographically-hashed
    Evaluation Experiment Realization.
    
    Contains all 9 required dimensions:
    1. Task generation timestamps
    2. Task characteristics
    3. Vehicle trajectories
    4. Mobility state
    5. Initial conditions
    6. RSU configuration
    7. Workload configuration
    8. Seed
    9. Geometry
    """
    realization_id: str
    geometry: str
    workload: int
    seed: int
    eval_seed: int
    
    # 1 & 2: Tasks (timestamps + characteristics)
    tasks: List[Dict[str, Any]]
    
    # 3: Vehicle trajectories
    vehicle_trajectories: List[Dict[str, Any]]
    
    # 4: Mobility state
    mobility_states: List[Dict[str, Any]]
    
    # 5: Initial conditions
    initial_conditions: Dict[str, Any]
    
    # 6: RSU configuration
    rsu_configurations: List[Dict[str, Any]]
    
    # 7: Workload configuration
    workload_configuration: Dict[str, Any]
    
    # Environment fingerprint and physics locks
    environment_configuration: Dict[str, Any]
    
    # Cryptographic SHA-256 hash of payload
    realization_hash: str = ""
    created_at: str = ""

    def compute_payload_dict(self) -> Dict[str, Any]:
        """Returns the canonical payload dictionary without the realization_hash field."""
        return {
            "realization_id": self.realization_id,
            "geometry": self.geometry,
            "workload": int(self.workload),
            "seed": int(self.seed),
            "eval_seed": int(self.eval_seed),
            "initial_conditions": self.initial_conditions,
            "rsu_configurations": self.rsu_configurations,
            "workload_configuration": self.workload_configuration,
            "environment_configuration": self.environment_configuration,
            "vehicle_trajectories": self.vehicle_trajectories,
            "mobility_states": self.mobility_states,
            "tasks": self.tasks,
        }

    def compute_hash(self) -> str:
        """Computes the deterministic cryptographic SHA-256 hash over canonical payload JSON."""
        payload = self.compute_payload_dict()
        canonical_json = json.dumps(payload, sort_keys=True, cls=RealizationEncoder, indent=2)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def verify_hash(self) -> bool:
        """Verifies that the stored realization_hash strictly matches the computed payload hash."""
        if not self.realization_hash:
            return False
        return self.compute_hash() == self.realization_hash

    def to_dict(self) -> Dict[str, Any]:
        """Converts complete realization to dictionary including realization_hash."""
        d = self.compute_payload_dict()
        d["realization_hash"] = self.realization_hash or self.compute_hash()
        d["created_at"] = self.created_at
        return d

    def save(self, filepath: str) -> str:
        """Saves realization to disk with updated hash."""
        self.realization_hash = self.compute_hash()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, sort_keys=True, cls=RealizationEncoder, indent=2)
        return self.realization_hash

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentRealization":
        """Reconstructs realization from dictionary."""
        return cls(
            realization_id=d["realization_id"],
            geometry=d["geometry"],
            workload=int(d["workload"]),
            seed=int(d["seed"]),
            eval_seed=int(d["eval_seed"]),
            tasks=d["tasks"],
            vehicle_trajectories=d["vehicle_trajectories"],
            mobility_states=d["mobility_states"],
            initial_conditions=d["initial_conditions"],
            rsu_configurations=d["rsu_configurations"],
            workload_configuration=d["workload_configuration"],
            environment_configuration=d["environment_configuration"],
            realization_hash=d.get("realization_hash", ""),
            created_at=d.get("created_at", "")
        )

    @classmethod
    def load(cls, filepath: str) -> "ExperimentRealization":
        """Loads realization from file."""
        with open(filepath, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)
