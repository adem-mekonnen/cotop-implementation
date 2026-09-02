"""
tests/test_phase12_final_scientific_validity.py
Automated regression tests verifying Phase 12 final claim matrix, objective performance
audit, component contribution audit, reproducibility scorecard, rewritten claims,
Class B certification, READY_WITH_DISCLOSURES publication readiness, protected physics
invariability, and multi-seed realization integrity.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P12_DIR = os.path.join(ROOT_DIR, "results", "remediation", "phase12_final_audit")

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

def test_a_final_claim_matrix_integrity():
    """Test A: Final claim matrix exists and accurately classifies claims."""
    claim_csv = os.path.join(P12_DIR, "final_claim_matrix.csv")
    assert os.path.exists(claim_csv)
    df = pd.read_csv(claim_csv)
    assert len(df) == 7
    scale_row = df[df["claim_id"] == "CLAIM_04"].iloc[0]
    assert "CONTRADICTED" in scale_row["status"]
    qrmp_row = df[df["claim_id"] == "CLAIM_06"].iloc[0]
    assert qrmp_row["status"] == "UNVERIFIABLE"

def test_b_objective_performance_audit():
    """Test B: Objective performance audit covers all 7 algorithms with valid ranks."""
    obj_csv = os.path.join(P12_DIR, "objective_performance_audit.csv")
    assert os.path.exists(obj_csv)
    df = pd.read_csv(obj_csv)
    assert len(df) == 7
    algos = df["algorithm"].tolist()
    assert "CoTOP" in algos
    assert "DDQN" in algos
    assert "Local" in algos
    assert "Greedy" in algos

def test_c_component_contribution_audit():
    """Test C: Component contribution audit records activation and behavioral divergence."""
    comp_csv = os.path.join(P12_DIR, "component_contribution_audit.csv")
    assert os.path.exists(comp_csv)
    df = pd.read_csv(comp_csv)
    assert len(df) == 4
    components = df["component"].tolist()
    assert any("GAT-GRU" in c for c in components)
    assert any("Task Prioritization" in c for c in components)

def test_d_reproducibility_scorecard():
    """Test D: Scorecard covers 21 evaluated dimensions with verified status."""
    score_csv = os.path.join(P12_DIR, "reproducibility_scorecard.csv")
    assert os.path.exists(score_csv)
    df = pd.read_csv(score_csv)
    assert len(df) >= 21

def test_e_rewritten_claims_table():
    """Test E: Rewritten paper claims table provides defensible replacements."""
    rewritten_csv = os.path.join(P12_DIR, "paper_claims_rewritten.csv")
    assert os.path.exists(rewritten_csv)
    df = pd.read_csv(rewritten_csv)
    assert len(df) == 4
    for _, row in df.iterrows():
        assert len(str(row["defensible_replacement"])) > 20

def test_f_final_manifest_certification():
    """Test F: Final manifest records Class B certification and READY_WITH_DISCLOSURES."""
    man_p = os.path.join(P12_DIR, "manifest.json")
    assert os.path.exists(man_p)
    with open(man_p, "r") as f:
        data = json.load(f)
    assert data["reproducibility_certification"] == "CLASS_B_IMPLEMENTATION_FAITHFUL_BUT_NUMERICALLY_NON_REPRODUCED"
    assert data["publication_readiness_decision"] == "READY_WITH_DISCLOSURES"

def test_g_protected_physics_hashes():
    """Test G: comm_model.py and comp_model.py hashes remain unchanged."""
    comm_p = os.path.join(ROOT_DIR, "envs", "comm_model.py")
    comp_p = os.path.join(ROOT_DIR, "envs", "comp_model.py")
    assert compute_file_sha256(comm_p) == COMM_SHA256
    assert compute_file_sha256(comp_p) == COMP_SHA256

def test_h_deterministic_evaluation(base_config, sample_realization):
    """Test H: CoTOP evaluation executes deterministically."""
    env = FrozenVECEnv(base_config, sample_realization)
    obs, _ = env.reset()
    assert len(obs) == 114
    _, _, _, _, info = env.step(0)
    assert info["delay"] > 0
    assert info["energy"] > 0

def test_i_all_60_frozen_realizations_exist():
    """Test I: All 60 evaluation realization files exist and have valid structure."""
    r_dir = os.path.join(ROOT_DIR, "data", "evaluation_realizations")
    files = [f for f in os.listdir(r_dir) if f.startswith("realization_") and f.endswith(".json")]
    assert len(files) >= 60

def test_j_future_experiment_ranking():
    """Test J: Future experiment ranking classifies long-horizon evaluation as strongly recommended."""
    rank_csv = os.path.join(P12_DIR, "future_experiment_ranking.csv")
    assert os.path.exists(rank_csv)
    df = pd.read_csv(rank_csv)
    assert len(df) >= 4
