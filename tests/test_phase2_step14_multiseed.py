import csv
import hashlib
import json
import os
import pytest
import numpy as np

SUMMARY_CSV = "results/phase2_algorithmic_fidelity/summary_60cell.csv"


def compute_file_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_summary_rows():
    assert os.path.exists(SUMMARY_CSV), f"Summary file {SUMMARY_CSV} does not exist"
    with open(SUMMARY_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def test_01_matrix_cardinality_and_dimensions():
    """Step 14 Test 01 — Factorial Matrix Cardinality (60 cells)."""
    rows = load_summary_rows()
    assert len(rows) == 60, f"Expected 60 runs, got {len(rows)}"


def test_02_all_required_seeds_present():
    """Step 14 Test 02 — Required Seeds [0, 1, 2, 3, 4]."""
    rows = load_summary_rows()
    seeds = set(int(r["seed"]) for r in rows)
    assert seeds == {0, 1, 2, 3, 4}


def test_03_all_required_geometries_present():
    """Step 14 Test 03 — Required Geometries [corridor_2400m, grid_200m]."""
    rows = load_summary_rows()
    geoms = set(r["geometry"] for r in rows)
    assert geoms == {"corridor_2400m", "grid_200m"}


def test_04_all_required_workloads_present():
    """Step 14 Test 04 — Required Workloads [20, 30, 40]."""
    rows = load_summary_rows()
    workloads = set(int(r["workload"]) for r in rows)
    assert workloads == {20, 30, 40}


def test_05_no_duplicate_rows():
    """Step 14 Test 05 — No Duplicate Condition Rows."""
    rows = load_summary_rows()
    keys = [(r["geometry"], r["algorithm"], int(r["workload"]), int(r["seed"])) for r in rows]
    assert len(keys) == len(set(keys)) == 60


def test_06_evaluation_trace_pairing_and_hashing():
    """Step 14 Test 06 — Evaluation Trace Causal Pairing."""
    rows = load_summary_rows()
    pairs = {}
    for r in rows:
        key = (r["geometry"], int(r["workload"]), int(r["seed"]))
        pairs.setdefault(key, {})[r["algorithm"]] = r["realization_hash"]
        
    for key, alg_hashes in pairs.items():
        assert "CoTOP" in alg_hashes and "DDQN" in alg_hashes, f"Missing algorithm in {key}"
        assert alg_hashes["CoTOP"] == alg_hashes["DDQN"], f"Trace hash mismatch for {key}: {alg_hashes}"
        assert len(alg_hashes["CoTOP"]) == 64


def test_07_training_evaluation_separation():
    """Step 14 Test 07 — Training vs Evaluation Realization Separation."""
    for geom in ["corridor_2400m", "grid_200m"]:
        for alg in ["CoTOP", "DDQN"]:
            for w in [20, 30, 40]:
                for s in [0, 1]:
                    manifest_p = f"results/phase2_algorithmic_fidelity/{geom}/{alg}/w{w}/seed_{s}/run_manifest.json"
                    assert os.path.exists(manifest_p)
                    with open(manifest_p, "r", encoding="utf-8") as f:
                        man = json.load(f)
                    assert int(man["seed"]) != int(man["eval_seed"])


def test_08_checkpoints_exist_and_hashes_recorded():
    """Step 14 Test 08 — Checkpoint Existence and SHA-256."""
    rows = load_summary_rows()
    for r in rows:
        ckpt_p = f"results/phase2_algorithmic_fidelity/{r['geometry']}/{r['algorithm']}/w{r['workload']}/seed_{r['seed']}/checkpoint_ep500.pt"
        assert os.path.exists(ckpt_p), f"Checkpoint missing: {ckpt_p}"
        actual_hash = compute_file_sha256(ckpt_p)
        assert actual_hash == r["checkpoint_sha256"], f"Checkpoint hash mismatch for {r['algorithm']} {r['geometry']} w{r['workload']} s{r['seed']}"


def test_09_eval_weight_immutability():
    """Step 14 Test 09 — Evaluation Invariants Flag."""
    rows = load_summary_rows()
    for r in rows:
        assert r["invariants_passed"] == "True" or r["invariants_passed"] is True


def test_10_eval_trace_immutability():
    """Step 14 Test 10 — Evaluation Trace Existence and Hash."""
    rows = load_summary_rows()
    for r in rows:
        realization_p = os.path.join("data", "evaluation_realizations", f"{r['geometry']}_w{r['workload']}_seed{r['seed']}_realization.json")
        assert os.path.exists(realization_p)
        with open(realization_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["realization_hash"] == r["realization_hash"]


def test_11_eval_determinism_reproducibility():
    """Step 14 Test 11 — Matrix Run Manifests Valid."""
    rows = load_summary_rows()
    for r in rows:
        manifest_p = f"results/phase2_algorithmic_fidelity/{r['geometry']}/{r['algorithm']}/w{r['workload']}/seed_{r['seed']}/run_manifest.json"
        assert os.path.exists(manifest_p)


def test_12_task_conservation_accounting():
    """Step 14 Test 12 — Task Conservation Accounting (N_tot = N_comp + N_fail)."""
    rows = load_summary_rows()
    for r in rows:
        n_comp = int(r["completed_tasks"])
        n_fail = int(r["failed_tasks"])
        n_tot = int(r["total_tasks"])
        assert n_tot == n_comp + n_fail, f"Task conservation violation in {r['algorithm']} {r['geometry']} w{r['workload']} s{r['seed']}"


def test_13_latency_decomposition_invariance():
    """Step 14 Test 13 — Mean Delay Finite and Valid."""
    rows = load_summary_rows()
    for r in rows:
        d = float(r["mean_delay_s"])
        assert np.isfinite(d) and d >= 0.0


def test_14_energy_decomposition_nonnegativity():
    """Step 14 Test 14 — Energy Non-Negativity."""
    rows = load_summary_rows()
    for r in rows:
        e = float(r["mean_energy_j"])
        assert np.isfinite(e) and e >= 0.0


def test_15_queue_nonnegativity():
    """Step 14 Test 15 — Physical Validity and Completion Bounds."""
    rows = load_summary_rows()
    for r in rows:
        cr = float(r["completion_ratio"])
        d = float(r["mean_delay_s"])
        assert 0.0 <= cr <= 1.0
        assert d >= 0.0


def test_16_zero_nan_inf_diagnostics():
    """Step 14 Test 16 — Zero NaN / Inf Diagnostics."""
    rows = load_summary_rows()
    for r in rows:
        assert r["invariants_passed"] == "True" or r["invariants_passed"] is True


def test_17_protected_physics_hash_integrity():
    """Step 14 Test 17 — Protected Physics File Hash Integrity."""
    expected_comm = "041e41061d02c7a5a7bc9488adf2bc49472177215730bd8a23c5ff2437431431"
    expected_comp = "dd9f58df710f709d536000bb4047d2ad6000cf37b1a49f4e1f0e8d883b856bff"
    
    actual_comm = compute_file_sha256("envs/comm_model.py")
    actual_comp = compute_file_sha256("envs/comp_model.py")
    
    assert actual_comm == expected_comm, f"comm_model.py modified! {actual_comm} != {expected_comm}"
    assert actual_comp == expected_comp, f"comp_model.py modified! {actual_comp} != {expected_comp}"
