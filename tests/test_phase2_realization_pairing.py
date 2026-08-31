"""
tests/test_phase2_realization_pairing.py

Comprehensive Test Suite for Controlled Experiment Realizations (Stage 7).
Verifies:
1. Realization integrity and cryptographic SHA-256 immutability.
2. Complete persistence of all 9 required dimensions.
3. Strict enforcement of all 5 rejection gates.
4. Paired consumption across CoTOP, DDQN, Greedy, and Local on the exact same realization.
"""

import os
import copy
import pytest
import numpy as np
import torch

from experiments.realizations.schema import ExperimentRealization
from experiments.realizations.generator import RealizationGenerator
from experiments.realizations.validator import (
    RealizationValidator,
    RealizationHashTamperedError,
    GeometryMismatchError,
    WorkloadMismatchError,
    SeedMismatchError,
    EnvironmentConfigMismatchError
)
from experiments.realizations.runner import RealizationRunner


@pytest.fixture
def sample_realization():
    """Generates a deterministic sample realization for unit testing."""
    generator = RealizationGenerator()
    realization = generator.generate_realization(
        geometry="corridor_2400m",
        workload=20,
        seed=42,
        eval_seed_offset=30000,
        num_vehicles=10
    )
    return realization


def test_01_realization_schema_and_hash_immutability(sample_realization):
    """
    Test 01 — Schema and Cryptographic Hash Immutability
    Verify that the realization serializes, deserializes, and verifies its SHA-256 hash.
    """
    real = sample_realization
    assert real.realization_hash != ""
    assert len(real.realization_hash) == 64
    assert real.verify_hash(), "Realization SHA-256 hash must verify against payload"
    
    # Test JSON roundtrip
    d = real.to_dict()
    reconstructed = ExperimentRealization.from_dict(d)
    assert reconstructed.verify_hash()
    assert reconstructed.realization_hash == real.realization_hash
    assert reconstructed.compute_hash() == real.compute_hash()


def test_02_all_nine_required_entities_present(sample_realization):
    """
    Test 02 — Completeness of 9 Persisted Dimensions
    Verify all 9 required entities are non-empty and well-structured:
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
    real = sample_realization
    
    # 1 & 2. Tasks (timestamps & characteristics)
    assert len(real.tasks) == 200  # 20 tasks * 10 vehicles
    for t in real.tasks:
        assert "task_id" in t
        assert "vehicle_id" in t
        assert "generation_timestamp" in t
        assert "size_rho" in t and 2.0e6 <= t["size_rho"] <= 5.0e6
        assert "cpu_phi" in t and 1.0e6 <= t["cpu_phi"] <= 10.0e6
        assert "max_delay_d" in t and 20.0 <= t["max_delay_d"] <= 30.0
        assert "priority_weight" in t and 0.0 <= t["priority_weight"] <= 1.0

    # 3. Vehicle trajectories
    assert len(real.vehicle_trajectories) == 10
    for vt in real.vehicle_trajectories:
        assert "vehicle_id" in vt
        assert "entry_time" in vt
        assert "initial_position" in vt
        assert "initial_speed" in vt
        assert len(vt["trajectory_points"]) > 0

    # 4. Mobility state
    assert len(real.mobility_states) == 10
    for ms in real.mobility_states:
        assert "vehicle_id" in ms
        assert "predicted_dwell_time_per_rsu" in ms
        assert len(ms["predicted_dwell_time_per_rsu"]) == 6

    # 5. Initial conditions
    assert "start_sim_time" in real.initial_conditions
    assert "num_vehicles" in real.initial_conditions and real.initial_conditions["num_vehicles"] == 10
    assert "active_vehicle_ids" in real.initial_conditions
    assert "initial_rsu_backlog_cycles" in real.initial_conditions

    # 6. RSU configurations
    assert len(real.rsu_configurations) == 6
    for rsu in real.rsu_configurations:
        assert "rsu_id" in rsu
        assert "location" in rsu
        assert "cpu_capacity_f" in rsu and rsu["cpu_capacity_f"] > 0
        assert "transmission_power_P_R" in rsu and rsu["transmission_power_P_R"] > 0
        assert "comm_range" in rsu and rsu["comm_range"] == 400.0

    # 7. Workload configuration
    assert real.workload_configuration["tasks_per_vehicle"] == 20
    assert real.workload_configuration["total_tasks"] == 200

    # 8. Seed
    assert real.seed == 42
    assert real.eval_seed == 30042

    # 9. Geometry
    assert real.geometry == "corridor_2400m"


def test_03_gate1_rejection_on_hash_tampering(sample_realization):
    """
    Test 03 — Gate 1 Rejection on Hash Tampering
    Tamper with a single byte in task data size and verify validator rejects.
    """
    real = copy.deepcopy(sample_realization)
    
    # Tamper with task 0 size by 1 byte
    real.tasks[0]["size_rho"] += 1.0
    
    with pytest.raises(RealizationHashTamperedError) as exc_info:
        RealizationValidator.validate(real)
    assert "GATE 1 REJECTION" in str(exc_info.value)


def test_04_gate2_rejection_on_geometry_mismatch(sample_realization):
    """
    Test 04 — Gate 2 Rejection on Geometry Mismatch
    Attempt to evaluate corridor_2400m realization in grid_200m experiment.
    """
    real = sample_realization
    with pytest.raises(GeometryMismatchError) as exc_info:
        RealizationValidator.validate(real, expected_geometry="grid_200m")
    assert "GATE 2 REJECTION" in str(exc_info.value)


def test_05_gate3_rejection_on_workload_mismatch(sample_realization):
    """
    Test 05 — Gate 3 Rejection on Workload Mismatch
    Attempt to run w30 experiment against w20 realization.
    """
    real = sample_realization
    with pytest.raises(WorkloadMismatchError) as exc_info:
        RealizationValidator.validate(real, expected_workload=30)
    assert "GATE 3 REJECTION" in str(exc_info.value)


def test_06_gate4_rejection_on_seed_mismatch(sample_realization):
    """
    Test 06 — Gate 4 Rejection on Seed Mismatch
    Attempt to run seed 99 against seed 42 realization.
    """
    real = sample_realization
    with pytest.raises(SeedMismatchError) as exc_info:
        RealizationValidator.validate(real, expected_seed=99)
    assert "GATE 4 REJECTION" in str(exc_info.value)


def test_07_gate5_rejection_on_environment_config_mismatch(sample_realization):
    """
    Test 07 — Gate 5 Rejection on Environment Fingerprint / Physics Mismatch
    """
    real = sample_realization
    with pytest.raises(EnvironmentConfigMismatchError) as exc_info:
        RealizationValidator.validate(real, expected_env_fingerprint="tampered_fake_fingerprint_00000")
    assert "GATE 5 REJECTION" in str(exc_info.value)


def test_08_four_way_controlled_paired_consumption(sample_realization):
    """
    Test 08 — 4-Way Controlled Paired Consumption
    Verify that CoTOP, DDQN, Greedy, and Local all consume the EXACT same realization,
    observe identical incoming tasks and channel states, and produce valid execution results.
    """
    real = sample_realization
    runner = RealizationRunner()
    
    # 1. Run Local Policy
    res_local = runner.run_algorithm("Local", realization=real)
    assert res_local.total_tasks == 200
    assert len(res_local.decisions) == 200
    assert all(d == 0 for d in res_local.decisions), "Local policy must always choose standalone action 0"
    
    # 2. Run Greedy Policy
    res_greedy = runner.run_algorithm("Greedy", realization=real)
    assert res_greedy.total_tasks == 200
    assert len(res_greedy.decisions) == 200
    
    # 3. Run DDQN Policy (untrained or fresh agent)
    res_ddqn = runner.run_algorithm("DDQN", realization=real)
    assert res_ddqn.total_tasks == 200
    assert len(res_ddqn.decisions) == 200
    
    # 4. Run CoTOP Policy (untrained or fresh model)
    res_cotop = runner.run_algorithm("CoTOP", realization=real)
    assert res_cotop.total_tasks == 200
    assert len(res_cotop.decisions) == 200
    
    # Verify that all 4 runs share the exact same realization metadata and cryptographic hash
    assert res_local.realization_hash == real.realization_hash
    assert res_greedy.realization_hash == real.realization_hash
    assert res_ddqn.realization_hash == real.realization_hash
    assert res_cotop.realization_hash == real.realization_hash
    
    # Verify metrics are finite and non-negative
    for res in [res_local, res_greedy, res_ddqn, res_cotop]:
        assert np.isfinite(res.mean_delay_s) and res.mean_delay_s > 0
        assert np.isfinite(res.mean_energy_j) and res.mean_energy_j > 0
        assert 0.0 <= res.completion_ratio <= 1.0
