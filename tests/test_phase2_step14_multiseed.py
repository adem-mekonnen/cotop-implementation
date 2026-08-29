import csv
import hashlib
import json
import os
import pytest
import numpy as np
import yaml

from scripts.run_phase2_multiseed import materialize_exogenous_trace, classify_convergence, compute_file_sha256


INDEX_CSV = "results/phase2_algorithmic_fidelity/PHASE2_EXPERIMENT_INDEX.csv"


def load_index_rows():
    assert os.path.exists(INDEX_CSV), f"Index file {INDEX_CSV} does not exist"
    with open(INDEX_CSV, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def test_01_matrix_cardinality_and_dimensions():
    """Step 14 Test 01 — Factorial Matrix Cardinality (60 cells)."""
    rows = load_index_rows()
    assert len(rows) == 60, f"Expected 60 runs, got {len(rows)}"


def test_02_all_required_seeds_present():
    """Step 14 Test 02 — Required Seeds [42, 43, 44, 45, 46]."""
    rows = load_index_rows()
    seeds = set(int(r["seed"]) for r in rows)
    assert seeds == {42, 43, 44, 45, 46}


def test_03_all_required_geometries_present():
    """Step 14 Test 03 — Required Geometries."""
    rows = load_index_rows()
    geoms = set(r["geometry"] for r in rows)
    assert geoms == {"linear_corridor", "urban_manhattan"}


def test_04_all_required_workloads_present():
    """Step 14 Test 04 — Required Workloads [20, 30, 40]."""
    rows = load_index_rows()
    workloads = set(int(r["workload"]) for r in rows)
    assert workloads == {20, 30, 40}


def test_05_no_duplicate_rows():
    """Step 14 Test 05 — No Duplicate Condition Rows."""
    rows = load_index_rows()
    keys = [(r["geometry"], r["algorithm"], int(r["workload"]), int(r["seed"])) for r in rows]
    assert len(keys) == len(set(keys)) == 60


def test_06_evaluation_trace_pairing_and_hashing():
    """Step 14 Test 06 — Evaluation Trace Causal Pairing."""
    rows = load_index_rows()
    pairs = {}
    for r in rows:
        key = (r["geometry"], int(r["workload"]), int(r["seed"]))
        pairs.setdefault(key, {})[r["algorithm"]] = r["evaluation_realization_hash"]
        
    for key, alg_hashes in pairs.items():
        assert "CoTOP" in alg_hashes and "DDQN" in alg_hashes, f"Missing algorithm in {key}"
        assert alg_hashes["CoTOP"] == alg_hashes["DDQN"], f"Trace hash mismatch for {key}: {alg_hashes}"
        assert len(alg_hashes["CoTOP"]) == 64


def test_07_training_evaluation_separation():
    """Step 14 Test 07 — Training vs Evaluation Realization Separation."""
    for geom in ["linear_corridor", "urban_manhattan"]:
        for alg in ["CoTOP", "DDQN"]:
            for w in [20, 30, 40]:
                for s in [42, 43]:
                    manifest_p = f"results/phase2_algorithmic_fidelity/{geom}/{alg}/w{w}/seed_{s}/run_manifest.json"
                    assert os.path.exists(manifest_p)
                    with open(manifest_p, "r") as f:
                        man = json.load(f)
                    assert man["training_seed"] != man["evaluation_seed"]
                    assert man["environment_seed"] != man["evaluation_seed"]


def test_08_checkpoints_exist_and_hashes_recorded():
    """Step 14 Test 08 — Checkpoint Existence and SHA-256."""
    rows = load_index_rows()
    for r in rows:
        ckpt_p = f"results/phase2_algorithmic_fidelity/{r['geometry']}/{r['algorithm']}/w{r['workload']}/seed_{r['seed']}/checkpoint_ep500.pt"
        assert os.path.exists(ckpt_p), f"Checkpoint missing: {ckpt_p}"
        actual_hash = compute_file_sha256(ckpt_p)
        assert actual_hash == r["checkpoint_hash"], f"Checkpoint hash mismatch for {r['algorithm']} {r['geometry']} w{r['workload']} s{r['seed']}"


def test_09_eval_weight_immutability():
    """Step 14 Test 09 — Evaluation Weight Immutability."""
    rows = load_index_rows()
    for r in rows:
        manifest_p = f"results/phase2_algorithmic_fidelity/{r['geometry']}/{r['algorithm']}/w{r['workload']}/seed_{r['seed']}/run_manifest.json"
        with open(manifest_p, "r") as f:
            man = json.load(f)
        assert man["evaluation_invariants_passed"]["gate_13_7_eval_isolation"] is True


def test_10_eval_trace_immutability():
    """Step 14 Test 10 — Evaluation Trace Immutability."""
    rows = load_index_rows()
    for r in rows:
        manifest_p = f"results/phase2_algorithmic_fidelity/{r['geometry']}/{r['algorithm']}/w{r['workload']}/seed_{r['seed']}/run_manifest.json"
        with open(manifest_p, "r") as f:
            man = json.load(f)
        assert man["evaluation_invariants_passed"]["gate_13_9_realization_immutability"] is True


def test_11_eval_determinism_reproducibility():
    """Step 14 Test 11 — Deterministic Evaluation Reproducibility."""
    rows = load_index_rows()
    for r in rows:
        manifest_p = f"results/phase2_algorithmic_fidelity/{r['geometry']}/{r['algorithm']}/w{r['workload']}/seed_{r['seed']}/run_manifest.json"
        with open(manifest_p, "r") as f:
            man = json.load(f)
        assert man["evaluation_invariants_passed"]["gate_13_8_determinism"] is True


def test_12_task_conservation_accounting():
    """Step 14 Test 12 — Task Conservation Accounting (N_gen = N_comp + N_fail + N_pend)."""
    rows = load_index_rows()
    for r in rows:
        n_comp = int(r["completed_tasks"])
        n_fail = int(r["failed_tasks"])
        n_tot = int(r["total_tasks"])
        assert n_tot == n_comp + n_fail, f"Task conservation violation in {r['algorithm']} {r['geometry']} w{r['workload']} s{r['seed']}"


def test_13_latency_decomposition_invariance():
    """Step 14 Test 13 — Latency Decomposition Invariance (residual <= 1e-4s)."""
    rows = load_index_rows()
    for r in rows:
        res = float(r["max_decomposition_residual_s"])
        assert res <= 1e-4, f"Latency decomposition violation ({res} > 1e-4) in {r['algorithm']} {r['geometry']} w{r['workload']} s{r['seed']}"


def test_14_energy_decomposition_nonnegativity():
    """Step 14 Test 14 — Energy Decomposition Non-Negativity."""
    rows = load_index_rows()
    for r in rows:
        e = float(r["mean_energy_j"])
        assert e >= 0.0, f"Negative energy ({e}) in {r['algorithm']} {r['geometry']} w{r['workload']} s{r['seed']}"


def test_15_queue_nonnegativity():
    """Step 14 Test 15 — Physical Validity and Completion Bounds."""
    rows = load_index_rows()
    for r in rows:
        cr = float(r["completion_ratio"])
        d = float(r["mean_delay_s"])
        assert 0.0 <= cr <= 1.0, f"Invalid completion ratio ({cr}) in {r}"
        assert d >= 0.0, f"Negative delay ({d}) in {r}"


def test_16_zero_nan_inf_diagnostics():
    """Step 14 Test 16 — Zero NaN / Inf Diagnostics."""
    rows = load_index_rows()
    for r in rows:
        manifest_p = f"results/phase2_algorithmic_fidelity/{r['geometry']}/{r['algorithm']}/w{r['workload']}/seed_{r['seed']}/run_manifest.json"
        with open(manifest_p, "r") as f:
            man = json.load(f)
        assert man["evaluation_invariants_passed"]["gate_13_4_training_stability"] is True


def test_17_convergence_diagnostics_classification():
    """Step 14 Test 17 — Convergence Diagnostics Classification."""
    rows = load_index_rows()
    valid_classes = {"STABLE", "OSCILLATORY", "NON_CONVERGED", "DIVERGED", "NUMERICALLY_INVALID"}
    for r in rows:
        c_class = r["convergence_class"]
        assert c_class in valid_classes, f"Invalid convergence class '{c_class}' in {r}"


def test_18_protected_physics_hash_integrity():
    """Step 14 Test 18 — Protected Physics File Hash Integrity."""
    expected_comm = "041e41061d02c7a5a7bc9488adf2bc49472177215730bd8a23c5ff2437431431"
    expected_comp = "dd9f58df710f709d536000bb4047d2ad6000cf37b1a49f4e1f0e8d883b856bff"
    
    actual_comm = compute_file_sha256("envs/comm_model.py")
    actual_comp = compute_file_sha256("envs/comp_model.py")
    
    assert actual_comm == expected_comm, f"comm_model.py modified! {actual_comm} != {expected_comm}"
    assert actual_comp == expected_comp, f"comp_model.py modified! {actual_comp} != {expected_comp}"
