"""
experiments/realizations/__init__.py

Controlled Experiment Realization System (Stage 7).
"""

from experiments.realizations.schema import (
    ExperimentRealization,
    TaskRealization,
    VehicleTrajectoryRealization,
    MobilityStateRealization,
    RSUConfigRealization,
    InitialConditionsRealization,
    WorkloadConfigRealization,
    EnvConfigRealization
)
from experiments.realizations.generator import RealizationGenerator
from experiments.realizations.validator import (
    RealizationValidator,
    RealizationError,
    RealizationHashTamperedError,
    GeometryMismatchError,
    WorkloadMismatchError,
    SeedMismatchError,
    EnvironmentConfigMismatchError
)
from experiments.realizations.runner import RealizationRunner, RealizationRunResult

__all__ = [
    "ExperimentRealization",
    "TaskRealization",
    "VehicleTrajectoryRealization",
    "MobilityStateRealization",
    "RSUConfigRealization",
    "InitialConditionsRealization",
    "WorkloadConfigRealization",
    "EnvConfigRealization",
    "RealizationGenerator",
    "RealizationValidator",
    "RealizationRunner",
    "RealizationRunResult",
    "RealizationError",
    "RealizationHashTamperedError",
    "GeometryMismatchError",
    "WorkloadMismatchError",
    "SeedMismatchError",
    "EnvironmentConfigMismatchError"
]
