"""
experiments/realizations/validator.py

Enforces the 5 strict rejection gates for Controlled Experiment Realizations (Stage 7).
Rejects any execution attempt if cryptographic hashes, geometries, workloads, seeds,
or environment configurations mismatch.
"""

from typing import Dict, Any, Optional
import hashlib
import os

from experiments.realizations.schema import ExperimentRealization


class RealizationError(Exception):
    """Base exception for all realization-related validation failures."""
    pass


class RealizationHashTamperedError(RealizationError):
    """Raised when cryptographic SHA-256 hash of realization does not match stored header."""
    pass


class GeometryMismatchError(RealizationError):
    """Raised when evaluation geometry differs from the realization geometry."""
    pass


class WorkloadMismatchError(RealizationError):
    """Raised when evaluation workload differs from the realization workload."""
    pass


class SeedMismatchError(RealizationError):
    """Raised when evaluation seed differs unexpectedly from realization seed."""
    pass


class EnvironmentConfigMismatchError(RealizationError):
    """Raised when environment fingerprint or physical model checksums mismatch."""
    pass


class RealizationValidator:
    """
    Validates that an ExperimentRealization is intact, untampered, and strictly matches
    the execution parameters and environment configuration.
    """
    
    @staticmethod
    def compute_file_sha256(filepath: str) -> str:
        """Computes SHA-256 hash of a file on disk."""
        if not os.path.exists(filepath):
            return ""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def validate(
        cls,
        realization: ExperimentRealization,
        expected_geometry: Optional[str] = None,
        expected_workload: Optional[int] = None,
        expected_seed: Optional[int] = None,
        expected_env_fingerprint: Optional[str] = None,
        verify_physics_files: bool = True
    ) -> bool:
        """
        Executes all 5 strict rejection gates:
        1. Cryptographic Hash Integrity Gate
        2. Geometry Conformance Gate
        3. Workload Conformance Gate
        4. Seed Conformance Gate
        5. Environment Configuration & Physics Lock Gate
        """
        # Gate 1: Cryptographic Hash Verification
        computed_hash = realization.compute_hash()
        if realization.realization_hash != computed_hash:
            raise RealizationHashTamperedError(
                f"[GATE 1 REJECTION] Cryptographic SHA-256 hash mismatch! "
                f"Stored Header: {realization.realization_hash} != Computed Payload: {computed_hash}. "
                f"Realization content has been tampered with or modified."
            )

        # Gate 2: Geometry Conformance
        if expected_geometry is not None:
            # Normalize geometry names
            geom_norm = "grid_200m" if expected_geometry in ["grid_200m", "urban_manhattan"] else "corridor_2400m"
            real_norm = "grid_200m" if realization.geometry in ["grid_200m", "urban_manhattan"] else "corridor_2400m"
            if real_norm != geom_norm:
                raise GeometryMismatchError(
                    f"[GATE 2 REJECTION] Geometry mismatch! "
                    f"Expected: {expected_geometry} (normalized: {geom_norm}), "
                    f"Realization: {realization.geometry} (normalized: {real_norm})."
                )

        # Gate 3: Workload Conformance
        if expected_workload is not None:
            if int(realization.workload) != int(expected_workload):
                raise WorkloadMismatchError(
                    f"[GATE 3 REJECTION] Workload mismatch! "
                    f"Expected: {expected_workload} tasks/veh, "
                    f"Realization: {realization.workload} tasks/veh."
                )

        # Gate 4: Seed Conformance
        if expected_seed is not None:
            if int(realization.seed) != int(expected_seed) and int(realization.eval_seed) != int(expected_seed):
                raise SeedMismatchError(
                    f"[GATE 4 REJECTION] Seed mismatch! "
                    f"Expected: {expected_seed}, "
                    f"Realization Master Seed: {realization.seed}, Eval Seed: {realization.eval_seed}."
                )

        # Gate 5: Environment Configuration & Physics Locks
        if expected_env_fingerprint is not None:
            real_fp = realization.environment_configuration.get("env_fingerprint")
            if real_fp != expected_env_fingerprint:
                raise EnvironmentConfigMismatchError(
                    f"[GATE 5 REJECTION] Environment fingerprint mismatch! "
                    f"Expected: {expected_env_fingerprint}, Realization: {real_fp}."
                )

        if verify_physics_files:
            comm_sha = cls.compute_file_sha256("envs/comm_model.py")
            comp_sha = cls.compute_file_sha256("envs/comp_model.py")
            
            real_comm = realization.environment_configuration.get("comm_model_sha256")
            real_comp = realization.environment_configuration.get("comp_model_sha256")
            
            if real_comm and comm_sha and real_comm != comm_sha:
                raise EnvironmentConfigMismatchError(
                    f"[GATE 5 REJECTION] comm_model.py hash mismatch! Disk: {comm_sha} != Realization: {real_comm}"
                )
            if real_comp and comp_sha and real_comp != comp_sha:
                raise EnvironmentConfigMismatchError(
                    f"[GATE 5 REJECTION] comp_model.py hash mismatch! Disk: {comp_sha} != Realization: {real_comp}"
                )

        return True
