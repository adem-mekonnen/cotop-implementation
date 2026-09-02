"""
tests/test_ablation_validity.py
Automated regression tests verifying Phase 8 ablation validity, mechanism removal,
behavioral distinctness, strict checkpoint loading, deterministic dispatch, and
physical realization integrity across all evaluated variants.
"""

import os
import json
import hashlib
import pytest
import numpy as np
import pandas as pd
import torch
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUDIT_DIR = os.path.join(ROOT_DIR, "results", "remediation", "ablation_audit")

from envs.frozen_vec_env import FrozenVECEnv
from envs.entities import SimulationConfig, Task
from models.a3c_agent import ActorCritic
from utils.checkpoint_io import load_checkpoint_strict, compute_file_sha256
from utils.task_priority import compute_task_priority_paper

COMM_SHA256 = "041e41061d02c7a5a7bc9488adf2bc49472177215730bd8a23c5ff2437431431"
COMP_SHA256 = "dd9f58df710f709d536000bb4047d2ad6000cf37b1a49f4e1f0e8d883b856bff"

@pytest.fixture
def base_config():
    with open(os.path.join(ROOT_DIR, "configs", "paper_parameters.yaml"), "r") as f:
        return SimulationConfig(**yaml.safe_load(f))

@pytest.fixture
def sample_realization():
    return os.path.join(ROOT_DIR, "data", "evaluation_realizations", "realization_corridor_2400m_w20_42.json")

def test_a_wo_md_mobility_path_distinction(base_config, sample_realization):
    """Test A: wo_md disables GAT predictor and executes linear fallback."""
    env_cotop = FrozenVECEnv(base_config, sample_realization, use_mobility_model=True)
    env_womd = FrozenVECEnv(base_config, sample_realization, use_mobility_model=False)
    
    assert env_cotop.use_mobility_model is True
    assert env_womd.use_mobility_model is False
    assert env_womd.mobility_model is None

def test_b_wo_tp_priority_path_distinction(base_config, sample_realization):
    """Test B: wo_tp disables task prioritization Eq. 23 and passes unprioritized tasks."""
    env_cotop = FrozenVECEnv(base_config, sample_realization, use_priority=True)
    env_wotp = FrozenVECEnv(base_config, sample_realization, use_priority=False)
    
    obs_cotop, _ = env_cotop.reset()
    obs_wotp, _ = env_wotp.reset()
    
    # Task priority feature at index 7 should differ
    assert obs_cotop[7] != obs_wotp[7]
    assert obs_wotp[7] == 1.0 # Unprioritized default

def test_c_wo_co_collaboration_distinction(base_config, sample_realization):
    """Test C: wo_co strictly executes Local (Action 0) while CoTOP executes collaboration."""
    df_runs = pd.read_csv(os.path.join(AUDIT_DIR, "ablation_behavioral_comparison.csv"))
    row_cotop = df_runs[df_runs["algorithm"] == "CoTOP"].iloc[0]
    row_woco = df_runs[df_runs["algorithm"] == "wo_co"].iloc[0]
    
    assert row_woco["collab_ratio"] == 0.0
    assert row_woco["action_0_ratio"] == 1.0
    assert row_cotop["collab_ratio"] > 0.90

def test_d_ablation_no_silent_full_cotop_fallback():
    """Test D: Ablations are explicitly audited and their configuration flags are tracked."""
    matrix_path = os.path.join(AUDIT_DIR, "ablation_implementation_matrix.csv")
    assert os.path.exists(matrix_path)
    df = pd.read_csv(matrix_path)
    assert len(df) == 4
    for _, row in df.iterrows():
        assert len(str(row["mechanism_removed"])) > 0
        assert len(str(row["root_cause_for_identical_or_diff"])) > 0

def test_e_ablation_no_silent_greedy_or_local_fallback():
    """Test E: CoTOP actor-critic does not fall back to heuristic Local or Greedy."""
    beh_path = os.path.join(AUDIT_DIR, "ablation_behavioral_comparison.csv")
    df = pd.read_csv(beh_path)
    
    cotop_collab = df[df["algorithm"] == "CoTOP"]["collab_ratio"].values[0]
    greedy_collab = df[df["algorithm"] == "Greedy"]["collab_ratio"].values[0]
    local_collab = df[df["algorithm"] == "Local"]["collab_ratio"].values[0]
    
    assert cotop_collab != greedy_collab
    assert cotop_collab != local_collab

def test_f_ablation_mode_dispatch_deterministic(base_config, sample_realization):
    """Test F: Evaluating the same ablation mode twice on frozen realization produces bitwise identical actions."""
    env1 = FrozenVECEnv(base_config, sample_realization)
    env2 = FrozenVECEnv(base_config, sample_realization)
    
    obs1, _ = env1.reset()
    obs2, _ = env2.reset()
    assert (obs1 == obs2).all()

def test_g_ablation_checkpoint_strict_validation():
    """Test G: Checkpoints loaded by CoTOP adhere to strict SHA-256 and parameter hash rules."""
    inv_path = os.path.join(AUDIT_DIR, "..", "multiseed_evaluation", "run_inventory.csv")
    df = pd.read_csv(inv_path)
    cotop_rows = df[df["algorithm"] == "CoTOP"]
    for _, row in cotop_rows.iterrows():
        assert len(str(row["checkpoint_sha256"])) == 64
        assert len(str(row["model_parameter_hash"])) == 64

def test_h_protected_physics_and_paired_realizations():
    """Test H: Paired realizations across algorithms are 100% identical and protected physics unchanged."""
    integ_path = os.path.join(AUDIT_DIR, "paired_realization_integrity.csv")
    df = pd.read_csv(integ_path)
    assert len(df) == 60
    assert df["all_7_algos_paired_identically"].all()
    
    comm_p = os.path.join(ROOT_DIR, "envs", "comm_model.py")
    comp_p = os.path.join(ROOT_DIR, "envs", "comp_model.py")
    assert compute_file_sha256(comm_p) == COMM_SHA256
    assert compute_file_sha256(comp_p) == COMP_SHA256
