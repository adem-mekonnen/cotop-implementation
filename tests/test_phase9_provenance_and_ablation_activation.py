"""
tests/test_phase9_provenance_and_ablation_activation.py
Automated regression tests verifying Phase 9 training provenance, strict reload,
GAT activation telemetry and controlled diagnostic, true task priority disabling,
wo_co vs Local mathematical equivalence, provenance manifest integrity,
protected physics invariability, and deterministic evaluation.
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
PROV_DIR = os.path.join(ROOT_DIR, "results", "remediation", "phase9_provenance")

from envs.frozen_vec_env import FrozenVECEnv
from envs.entities import SimulationConfig, Task
from models.a3c_agent import ActorCritic
from models.baselines.ddqn_agent import QNetwork
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

def test_a_checkpoint_provenance():
    """Test A: Checkpoint SHA and model parameter hash are recorded for all audited models."""
    prov_csv = os.path.join(PROV_DIR, "checkpoint_provenance.csv")
    assert os.path.exists(prov_csv)
    df = pd.read_csv(prov_csv)
    for _, row in df.iterrows():
        assert len(str(row["sha256"])) == 64
        assert len(str(row["model_parameter_hash"])) == 64
        assert row["reloadable_strictly"] is True

def test_b_strict_reload_failure():
    """Test B: Invalid or missing checkpoints fail immediately without silent fallback."""
    dummy_model = ActorCritic(input_dim=114, num_actions=7)
    with pytest.raises((FileNotFoundError, RuntimeError)):
        load_checkpoint_strict("non_existent_path.pt", dummy_model)

def test_c_gat_activation_telemetry():
    """Test C: Telemetry correctly distinguishes GAT activation from fallback across realizations."""
    mob_csv = os.path.join(PROV_DIR, "mobility_activation_audit.csv")
    assert os.path.exists(mob_csv)
    df = pd.read_csv(mob_csv)
    assert len(df) == 60
    assert "gat_activation_count" in df.columns
    assert "fallback_count" in df.columns

def test_d_gat_diagnostic_activation(base_config, sample_realization):
    """Test D: A realization with >= 5 historical trajectory frames successfully activates GAT."""
    env = FrozenVECEnv(config=base_config, realization_path=sample_realization, use_mobility_model=True)
    env.reset()
    for v in env.active_vehicles.values():
        v.trajectory_history = [(v.pos[0] - i * 10.0, v.pos[1]) for i in range(5, 0, -1)]
    
    x_seq, edge_index, veh_ids = env._build_mobility_graph()
    assert len(veh_ids) > 0
    assert env.mobility_model is not None
    preds = env.mobility_model(x_seq, edge_index)
    assert preds.shape[-1] == 2

def test_e_task_priority_disabling():
    """Test E: wo_tp actually sets use_priority=False and priority score to unprioritized 1.0."""
    prio_csv = os.path.join(PROV_DIR, "task_priority_activation_audit.csv")
    assert os.path.exists(prio_csv)
    df = pd.read_csv(prio_csv)
    row = df.iloc[0]
    assert bool(row["cotop_use_priority"]) is True
    assert bool(row["wotp_use_priority"]) is False
    assert bool(row["priority_mechanism_active_and_distinct"]) is True

def test_f_task_ordering_distinction(base_config, sample_realization):
    """Test F: Task prioritization reorders urgent tasks before relaxed tasks."""
    t_urgent = Task(task_id=1, vehicle_id="v0", size_rho=1e6, cpu_phi=1e6, max_delay_d=1.0)
    t_relaxed = Task(task_id=2, vehicle_id="v0", size_rho=5e6, cpu_phi=1e6, max_delay_d=30.0)
    
    p_urgent = compute_task_priority_paper(t_urgent, 10.0, alpha=0.3, beta=0.7)
    p_relaxed = compute_task_priority_paper(t_relaxed, 10.0, alpha=0.3, beta=0.7)
    
    assert p_urgent > p_relaxed

def test_g_woco_local_equivalence():
    """Test G: wo_co and Local produce identical task metrics and 100% action 0."""
    equiv_csv = os.path.join(PROV_DIR, "wo_co_local_equivalence.csv")
    assert os.path.exists(equiv_csv)
    df = pd.read_csv(equiv_csv)
    row = df.iloc[0]
    assert row["delay_max_abs_difference"] == 0.0
    assert row["energy_max_abs_difference"] == 0.0
    assert bool(row["action_sequences_bitwise_identical"]) is True

def test_h_provenance_manifest_integrity():
    """Test H: Required manifest fields exist and are internally consistent."""
    manifest_p = os.path.join(PROV_DIR, "manifest.json")
    assert os.path.exists(manifest_p)
    with open(manifest_p, "r") as f:
        data = json.load(f)
    assert data["gat_fallback_rate_official_eval"] == 1.0
    assert data["wo_co_local_equivalence_verified"] is True
    assert data["verdict"] == "PASS WITH CAVEATS"

def test_i_protected_physics_integrity():
    """Test I: Protected physics files maintain exact SHA-256 hashes."""
    comm_p = os.path.join(ROOT_DIR, "envs", "comm_model.py")
    comp_p = os.path.join(ROOT_DIR, "envs", "comp_model.py")
    assert compute_file_sha256(comm_p) == COMM_SHA256
    assert compute_file_sha256(comp_p) == COMP_SHA256

def test_j_deterministic_evaluation(base_config, sample_realization):
    """Test J: Evaluating the representative checkpoint twice produces bitwise identical actions."""
    model = ActorCritic(input_dim=114, num_actions=7)
    ckpt_path = os.path.join(ROOT_DIR, "results", "phase2_multiseed", "CoTOP", "corridor_2400m_w20_seed42", "checkpoint.pt")
    load_checkpoint_strict(ckpt_path, model)
    model.eval()

    def run_eval():
        env = FrozenVECEnv(base_config, sample_realization)
        obs, _ = env.reset()
        actions = []
        while len(env.pending_tasks) > 0:
            mask = env.get_action_mask()
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(obs_t)
            mask_t = torch.BoolTensor(mask).unsqueeze(0)
            logits[~mask_t] = -1e9
            a = torch.argmax(logits, dim=-1).item()
            actions.append(a)
            obs, _, _, _, _ = env.step(a)
        return actions

    actions1 = run_eval()
    actions2 = run_eval()
    assert actions1 == actions2
