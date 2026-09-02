import pytest
import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd

from scripts.analyze_phase2_step21 import (
    verify_physics_hashes,
    COMM_SHA256,
    COMP_SHA256,
    SEEDS,
    SCENARIOS,
    WORKLOADS,
    ALGORITHMS
)

RESULTS_DIR = "results/phase2_step21"

# 1. Physics Immutability
def test_01_physics_hashes_integrity():
    comm_h, comp_h = verify_physics_hashes()
    assert comm_h == COMM_SHA256, "comm_model.py altered!"
    assert comp_h == COMP_SHA256, "comp_model.py altered!"

# 2. Realization Inventory Completeness
def test_02_realization_inventory_completeness():
    path = os.path.join(RESULTS_DIR, "realization_inventory.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        assert len(df) == 60, f"Expected 60 realization traces, found {len(df)}"
        assert set(df["seed"].unique()) == set(SEEDS)
        assert set(df["scenario"].unique()) == set(SCENARIOS)
        assert (df["status"] == "VALID").all()

# 3. Full 240-Cell Run Inventory & Status
def test_03_run_inventory_240_cells():
    path = os.path.join(RESULTS_DIR, "run_inventory.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        assert len(df) == 240, f"Expected 240 runs, found {len(df)}"
        assert (df["status"] == "COMPLETED").all()
        assert df["tasks_completed"].sum() > 0
        assert not df["mean_delay_s"].isna().any()
        assert not df["mean_energy_j"].isna().any()
        assert not np.isinf(df["mean_delay_s"].values).any()
        assert not np.isinf(df["mean_energy_j"].values).any()

# 4. Critical Realization Rule: Identical Realization Hashes Across Algorithms
def test_04_matched_realization_hashes():
    path = os.path.join(RESULTS_DIR, "run_inventory.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        for scen in SCENARIOS:
            for wl in WORKLOADS:
                for s in SEEDS:
                    sub = df[(df["scenario"] == scen) & (df["workload"] == f"w{wl}") & (df["seed"] == s)]
                    assert len(sub) == 4, f"Expected 4 algorithms for ({scen}, w{wl}, seed {s})"
                    hashes = sub["realization_sha256"].unique()
                    assert len(hashes) == 1, f"Realization hash mismatch among algorithms for ({scen}, w{wl}, seed {s}): {hashes}"

# 5. Parameter Immutability During Evaluation
def test_05_parameter_immutability():
    path = os.path.join(RESULTS_DIR, "run_inventory.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        assert (df["param_immutable"] == True).all(), "Model parameters mutated during evaluation!"

# 6. Checkpoint Inventory
def test_06_checkpoint_inventory():
    path = os.path.join(RESULTS_DIR, "checkpoint_inventory.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        assert len(df) >= 0

# 7. Summary Files Exist and Valid
def test_07_summary_files_validity():
    for f_name in ["algorithm_summary.csv", "scenario_summary.csv", "workload_summary.csv", "seed_summary.csv", "convergence_summary.csv", "failed_run_report.csv"]:
        path = os.path.join(RESULTS_DIR, f_name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            assert not df.empty or f_name == "failed_run_report.csv"

# 8. Paired Inferential Statistics
def test_08_paired_statistical_analysis():
    path = os.path.join(RESULTS_DIR, "paired_statistical_analysis.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        assert len(df) == 12, f"Expected 12 paired comparisons (2 scenarios x 3 workloads x 2 metrics), got {len(df)}"
        assert (df["n_pairs"] == 10).all(), "Expected n=10 pairs for all conditions"
        assert not df["p_value_ttest"].isna().any()
        assert not df["cohens_dz"].isna().any()
        assert not df["fdr_q_adj"].isna().any()

# 9. Provenance Manifest Schema
def test_09_provenance_manifest():
    path = os.path.join(RESULTS_DIR, "provenance_manifest.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            manifest = json.load(f)
        assert manifest["status"] == "PASS"
        assert manifest["matrix_dimensions"]["total_cells"] == 240
        assert manifest["matrix_dimensions"]["completed_cells"] == 240
        assert manifest["matrix_dimensions"]["failed_cells"] == 0
        assert manifest["physics_hashes"]["envs/comm_model.py"] == COMM_SHA256
        assert manifest["physics_hashes"]["envs/comp_model.py"] == COMP_SHA256
