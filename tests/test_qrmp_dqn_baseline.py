"""
tests/test_qrmp_dqn_baseline.py
Automated regression tests verifying Phase 11 QRMP-DQN specification, strict checkpoint
rejection on missing/corrupted files, action space validity, task-level accounting,
deterministic evaluation, frozen realization integrity, and strict absence of silent
algorithm substitution (explicit reporting of baseline exclusion).
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
import torch
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QRMP_DIR = os.path.join(ROOT_DIR, "results", "remediation", "qrmp_dqn_audit")

from envs.frozen_vec_env import FrozenVECEnv
from envs.entities import SimulationConfig
from models.a3c_agent import ActorCritic
from utils.checkpoint_io import load_checkpoint_strict, compute_file_sha256

COMM_SHA256 = "041e41061d02c7a5a7bc9488adf2bc49472177215730bd8a23c5ff2437431431"
COMP_SHA256 = "dd9f58df710f709d536000bb4047d2ad6000cf37b1a49f4e1f0e8d883b856bff"

@pytest.fixture
def base_config():
    with open(os.path.join(ROOT_DIR, "configs", "paper_parameters.yaml"), "r") as f:
        return SimulationConfig(**yaml.safe_load(f))

@pytest.fixture
def sample_realization():
    return os.path.join(ROOT_DIR, "data", "evaluation_realizations", "realization_corridor_2400m_w20_42.json")

def test_a_qrmp_specification_and_status():
    """Test A: QRMP specification exists and records NOT_REPRODUCIBLE_FROM_AVAILABLE_EVIDENCE status."""
    spec_p = os.path.join(QRMP_DIR, "qrmp_specification.json")
    assert os.path.exists(spec_p)
    with open(spec_p, "r") as f:
        data = json.load(f)
    assert data["scientific_status"] == "NOT_REPRODUCIBLE_FROM_AVAILABLE_EVIDENCE"
    assert "STAR-RIS" in data["exclusion_reason"]

def test_b_strict_checkpoint_loading_rejection():
    """Test B: Strict checkpoint loader rejects non-existent checkpoint path."""
    model = ActorCritic(input_dim=114, num_actions=7)
    with pytest.raises(FileNotFoundError):
        load_checkpoint_strict("results/checkpoints/non_existent_qrmp.pt", model)

def test_c_missing_checkpoint_fails_without_fallback():
    """Test C: Attempting to load missing QRMP checkpoint does not silently create random weights."""
    model = ActorCritic(input_dim=114, num_actions=7)
    with pytest.raises(FileNotFoundError):
        load_checkpoint_strict("missing_qrmp_checkpoint.pt", model)

def test_d_corrupted_checkpoint_fails(tmp_path):
    """Test D: Corrupted checkpoint fails with RuntimeError or EOFError."""
    corrupted_p = tmp_path / "corrupted_qrmp.pt"
    corrupted_p.write_bytes(b"CORRUPTED_TORCH_DATA_BYTES")
    model = ActorCritic(input_dim=114, num_actions=7)
    with pytest.raises((RuntimeError, EOFError, Exception)):
        load_checkpoint_strict(str(corrupted_p), model)

def test_e_incompatible_checkpoint_fails():
    """Test E: Incompatible model architecture causes strict checkpoint rejection."""
    valid_cotop_p = os.path.join(ROOT_DIR, "results", "phase2_multiseed", "CoTOP", "corridor_2400m_w20_seed42", "checkpoint.pt")
    model = ActorCritic(input_dim=154, num_actions=7)  # Incompatible dimension (154 vs 114)
    with pytest.raises(RuntimeError):
        load_checkpoint_strict(valid_cotop_p, model)

def test_f_action_space_validity(base_config, sample_realization):
    """Test F: Environment action mask has shape (7,) and valid boolean entries."""
    env = FrozenVECEnv(base_config, sample_realization)
    obs, _ = env.reset()
    mask = env.get_action_mask()
    assert mask.shape == (7,)
    assert mask.dtype == bool
    assert mask[0] is True or mask[0] == True

def test_g_task_level_accounting_integrity(base_config, sample_realization):
    """Test G: Task-level accounting guarantees completed + failed == total tasks (200)."""
    env = FrozenVECEnv(base_config, sample_realization)
    env.reset()
    while len(env.pending_tasks) > 0:
        env.step(0)
    assert len(env.completed_tasks) + len(env.failed_tasks) == 200

def test_h_frozen_realization_deterministic_eval(base_config, sample_realization):
    """Test H: Evaluating the frozen environment with fixed actions is bitwise deterministic."""
    def run_eval():
        env = FrozenVECEnv(base_config, sample_realization)
        env.reset()
        delays = []
        while len(env.pending_tasks) > 0:
            _, _, _, _, info = env.step(0)
            delays.append(info["delay"])
        return delays
    d1 = run_eval()
    d2 = run_eval()
    assert d1 == d2

def test_i_realization_hash_integrity(sample_realization):
    """Test I: Frozen evaluation realization contains valid metadata and hash fields."""
    with open(sample_realization, "r") as f:
        data = json.load(f)
    assert "hash" in data
    assert len(data["hash"]) == 64

def test_j_no_silent_algorithm_substitution():
    """Test J: System explicitly reports QRMP-DQN as excluded and does not substitute DDQN."""
    fid_csv = os.path.join(QRMP_DIR, "implementation_fidelity.csv")
    assert os.path.exists(fid_csv)
    df = pd.read_csv(fid_csv)
    qrmp_row = df[df["algorithm"] == "QRMP-DQN"].iloc[0]
    assert qrmp_row["evaluation_runs"] == 0
    assert qrmp_row["scientific_verdict"] == "NOT REPRODUCIBLE (FORMALLY EXCLUDED)"
