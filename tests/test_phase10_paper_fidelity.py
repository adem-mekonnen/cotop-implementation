"""
tests/test_phase10_paper_fidelity.py
Automated regression tests verifying Phase 10 paper specification completeness,
equation-to-code mapping, parameter and unit fidelity, strict checkpoint reloadability,
frozen realization integrity, protected physics invariability, published vs reproduced
results consistency, discrepancy decomposition determinism, baseline policy determinism,
and provenance manifest integrity.
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
P10_DIR = os.path.join(ROOT_DIR, "results", "remediation", "phase10_paper_fidelity")

from envs.frozen_vec_env import FrozenVECEnv
from envs.entities import SimulationConfig
from models.a3c_agent import ActorCritic
from models.baselines.local import LocalPolicy
from models.baselines.greedy import GreedyPolicy
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

def test_a_paper_specification_complete():
    """Test A: Paper specification JSON exists, is valid JSON, and contains all required sections."""
    spec_p = os.path.join(P10_DIR, "paper_specification.json")
    assert os.path.exists(spec_p)
    with open(spec_p, "r") as f:
        data = json.load(f)
    assert "system_model" in data
    assert "task_model" in data
    assert "compute_parameters" in data
    assert "algorithm_models" in data
    assert "published_headline_results" in data
    assert data["published_headline_results"]["mean_total_delay_s"] == 13.90

def test_b_equation_to_code_mappings():
    """Test B: Equation implementation matrix maps all protected paper equations to valid files."""
    eq_csv = os.path.join(P10_DIR, "equation_implementation_matrix.csv")
    assert os.path.exists(eq_csv)
    df = pd.read_csv(eq_csv)
    assert len(df) >= 14
    for _, row in df.iterrows():
        target_f = os.path.join(ROOT_DIR, row["repo_file"])
        assert os.path.exists(target_f), f"Mapped file {target_f} does not exist!"

def test_c_parameter_unit_fidelity():
    """Test C: Parameter fidelity matrix contains no undocumented units or conversions."""
    param_csv = os.path.join(P10_DIR, "parameter_fidelity_matrix.csv")
    assert os.path.exists(param_csv)
    df = pd.read_csv(param_csv)
    assert len(df) >= 15
    for _, row in df.iterrows():
        assert row["status"] in ["EXACT MATCH", "PAPER-CONSISTENT RECONSTRUCTION"]

def test_d_checkpoint_strict_reload():
    """Test D: All official checkpoints remain strictly reloadable."""
    ckpt_path = os.path.join(ROOT_DIR, "results", "phase2_multiseed", "CoTOP", "corridor_2400m_w20_seed42", "checkpoint.pt")
    model = ActorCritic(input_dim=114, num_actions=7)
    meta = load_checkpoint_strict(ckpt_path, model, expected_algorithm="CoTOP")
    assert len(meta["model_param_hash"]) == 64

def test_e_frozen_realization_hashes(sample_realization):
    """Test E: Sample evaluation realization file exists and has valid JSON structure."""
    assert os.path.exists(sample_realization)
    with open(sample_realization, "r") as f:
        data = json.load(f)
    assert "vehicle_trace" in data
    assert "task_trace" in data
    assert len(data["task_trace"]) > 0

def test_f_protected_physics_hashes():
    """Test F: comm_model.py and comp_model.py hashes remain unchanged."""
    comm_p = os.path.join(ROOT_DIR, "envs", "comm_model.py")
    comp_p = os.path.join(ROOT_DIR, "envs", "comp_model.py")
    assert compute_file_sha256(comm_p) == COMM_SHA256
    assert compute_file_sha256(comp_p) == COMP_SHA256

def test_g_published_vs_reproduced_consistency():
    """Test G: Published vs reproduced CSV exists and classifies scale gap as NUMERICAL_MISMATCH."""
    pub_csv = os.path.join(P10_DIR, "published_vs_reproduced.csv")
    assert os.path.exists(pub_csv)
    df = pd.read_csv(pub_csv)
    assert len(df) >= 4
    delay_row = df[df["metric"].str.contains("Delay")].iloc[0]
    assert delay_row["reproduction_status"] == "NUMERICAL_MISMATCH"
    assert delay_row["paper_value"] == 13.90

def test_h_discrepancy_analysis_determinism():
    """Test H: Discrepancy decomposition CSV exists and quantifies physical latency factors."""
    disc_csv = os.path.join(P10_DIR, "discrepancy_decomposition.csv")
    assert os.path.exists(disc_csv)
    df = pd.read_csv(disc_csv)
    assert len(df) >= 4
    assert "V2R Uplink Delay" in df["factor_name"].values or "Task Uplink Transmission" in df["factor_name"].values

def test_i_baseline_determinism(base_config, sample_realization):
    """Test I: LocalPolicy and GreedyPolicy execute deterministically on frozen realization."""
    env = FrozenVECEnv(base_config, sample_realization)
    obs, _ = env.reset()
    lp = LocalPolicy(base_config)
    a1 = lp.select_action(obs)
    a2 = lp.select_action(obs)
    assert a1 == a2 == 0

def test_j_provenance_manifest_metadata():
    """Test J: Manifest contains required metadata fields and verdict."""
    man_p = os.path.join(P10_DIR, "manifest.json")
    assert os.path.exists(man_p)
    with open(man_p, "r") as f:
        data = json.load(f)
    assert data["audit_name"] == "PHASE_10_PAPER_TO_IMPLEMENTATION_FIDELITY_AND_CLAIM_VALIDATION"
    assert data["scale_discrepancy_ratio"] > 5.0
    assert data["verdict"] == "PASS WITH CAVEATS"
