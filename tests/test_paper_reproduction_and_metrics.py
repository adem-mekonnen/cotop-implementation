"""
tests/test_paper_reproduction_and_metrics.py
Automated regression tests verifying metric aggregation determinism, denominator integrity,
explicit failure classifications, physical unit consistency, and run inventory provenance.
"""

import os
import json
import hashlib
import pytest
import numpy as np
import pandas as pd
import yaml
import torch

from envs.frozen_vec_env import FrozenVECEnv
from envs.entities import SimulationConfig
from models.baselines.local import LocalPolicy
from models.baselines.greedy import GreedyPolicy

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "paper_parameters.yaml")
REALIZATION_PATH = os.path.join(
    ROOT_DIR, "data", "evaluation_realizations", "realization_corridor_2400m_w20_seed42.json"
)

@pytest.fixture
def base_config():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["num_tasks_per_vehicle_range"] = [20, 20]
    return SimulationConfig(**cfg)

def test_a_metric_aggregation_determinism(base_config):
    """Test A: Metric aggregation is strictly deterministic given fixed inputs."""
    env = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    obs, _ = env.reset(seed=42)
    
    delays1, energies1 = [], []
    while len(env.pending_tasks) > 0:
        _, _, _, _, info = env.step(0)
        delays1.append(info["delay"])
        energies1.append(info["energy"])
    env.close()
    
    env2 = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    obs2, _ = env2.reset(seed=42)
    delays2, energies2 = [], []
    while len(env2.pending_tasks) > 0:
        _, _, _, _, info2 = env2.step(0)
        delays2.append(info2["delay"])
        energies2.append(info2["energy"])
    env2.close()
    
    assert np.allclose(delays1, delays2), "Delays are not deterministic across runs!"
    assert np.allclose(energies1, energies2), "Energies are not deterministic across runs!"
    assert np.mean(delays1) == np.mean(delays2)
    assert np.mean(energies1) == np.mean(energies2)

def test_b_completion_denominator_integrity(base_config):
    """Test B: Completion ratio denominator is strictly equal to total tasks generated."""
    env = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    env.reset(seed=42)
    
    comp_count = 0
    total_count = 0
    while len(env.pending_tasks) > 0:
        _, _, _, _, info = env.step(0)
        if info["completed"]:
            comp_count += 1
        total_count += 1
    env.close()
    
    assert total_count == 200, f"Expected 200 total tasks, got {total_count}"
    assert comp_count == 193, f"Expected 193 completed tasks, got {comp_count}"
    ratio = comp_count / total_count
    assert abs(ratio - 0.965) < 1e-6

def test_c_failure_classification_explicit(base_config):
    """Test C: Failed tasks have explicit non-null failure reason string."""
    env = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    env.reset(seed=42)
    
    reasons = []
    while len(env.pending_tasks) > 0:
        _, _, _, _, info = env.step(0)
        if not info["completed"]:
            reasons.append(info["failure_reason"])
    env.close()
    
    assert len(reasons) == 7, f"Expected 7 failures, got {len(reasons)}"
    assert all(r in ["COVERAGE_VIOLATION", "DEADLINE_EXCEEDED", "DUAL_VIOLATION"] for r in reasons)
    assert all(r == "COVERAGE_VIOLATION" for r in reasons)

def test_d_physical_units_validity(base_config):
    """Test D: Physical units are positive and scale within physical bounds."""
    env = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    env.reset(seed=42)
    
    while len(env.pending_tasks) > 0:
        _, _, _, _, info = env.step(0)
        assert 0.0 < info["delay"] < 100.0, f"Unphysical delay: {info['delay']} s"
        assert 0.0 < info["energy"] < 1000.0, f"Unphysical energy: {info['energy']} J"
        assert 0.0 <= info["comm_delay"] <= info["delay"]
        assert 0.0 <= info["comp_delay"] <= info["delay"]
    env.close()

def test_e_same_realization_reproducibility(base_config):
    """Test E: Exact same realization evaluated twice produces bit-for-bit identical hashes."""
    policy = GreedyPolicy(config=base_config)
    
    env = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    obs, _ = env.reset(seed=42)
    actions1 = []
    while len(env.pending_tasks) > 0:
        act = policy.select_action(obs)
        actions1.append(act)
        obs, _, _, _, _ = env.step(act)
    env.close()
    
    env2 = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    obs2, _ = env2.reset(seed=42)
    actions2 = []
    while len(env2.pending_tasks) > 0:
        act2 = policy.select_action(obs2)
        actions2.append(act2)
        obs2, _, _, _, _ = env2.step(act2)
    env2.close()
    
    assert actions1 == actions2, "Greedy policy produced non-deterministic action sequence!"
    hash1 = hashlib.sha256(json.dumps(actions1).encode()).hexdigest()
    hash2 = hashlib.sha256(json.dumps(actions2).encode()).hexdigest()
    assert hash1 == hash2

def test_f_checkpoint_provenance_record():
    """Test F: Checkpoint manifest exists and points to valid file with matching SHA-256."""
    manifest_path = os.path.join(
        ROOT_DIR, "results", "remediation", "training_pipeline_audit", "checkpoint_manifest.json"
    )
    assert os.path.exists(manifest_path), "Checkpoint manifest missing!"
    
    with open(manifest_path, "r") as f:
        data = json.load(f)
        
    ckpt_rel_path = data["checkpoint_path"]
    ckpt_full_path = os.path.join(ROOT_DIR, ckpt_rel_path)
    assert os.path.exists(ckpt_full_path), f"Checkpoint file not found: {ckpt_full_path}"
    
    with open(ckpt_full_path, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()
    assert actual_sha == data["checkpoint_sha256"], "Manifest SHA-256 does not match checkpoint file!"

def test_g_run_inventory_schema():
    """Test G: Run inventory CSV contains all required columns and valid records."""
    inv_path = os.path.join(
        ROOT_DIR, "results", "remediation", "paper_reproduction", "run_inventory.csv"
    )
    assert os.path.exists(inv_path), "Run inventory CSV missing!"
    
    df = pd.read_csv(inv_path)
    required_cols = [
        "run_id", "algorithm", "scenario", "workload", "seed",
        "mean_delay_s", "mean_energy_j", "completion_ratio",
        "tasks_generated", "tasks_completed", "tasks_failed",
        "checkpoint_sha256", "realization_sha256", "git_commit"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column in run_inventory.csv: {col}"
    assert len(df) >= 1, "Run inventory has no records!"
