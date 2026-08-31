import os
import json
import hashlib
import pytest
import numpy as np
import pandas as pd
import torch

from envs.frozen_vec_env import FrozenVECEnv
from envs.entities import SimulationConfig
from models.baselines.ddqn_agent import DDQNAgent

PROTECTED_COMM_HASH = "041e41061d02c7a5a7bc9488adf2bc49472177215730bd8a23c5ff2437431431"
PROTECTED_COMP_HASH = "dd9f58df710f709d536000bb4047d2ad6000cf37b1a49f4e1f0e8d883b856bff"

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def test_physics_immutability():
    """Verify protected communication and computation models have not been modified."""
    comm_hash = compute_sha256("envs/comm_model.py")
    comp_hash = compute_sha256("envs/comp_model.py")
    assert comm_hash == PROTECTED_COMM_HASH, f"comm_model hash mismatch: {comm_hash}"
    assert comp_hash == PROTECTED_COMP_HASH, f"comp_model hash mismatch: {comp_hash}"

def test_step14_artifact_manifest_completeness():
    """Verify that all required Step 14 artifacts exist for all 5 seeds."""
    seeds = [42, 43, 44, 45, 46]
    base_dir = "results/phase2_step14/linear_corridor_DDQN_w20"
    
    assert os.path.exists("results/phase2_step14/step14_seed_summary.csv")
    assert os.path.exists("results/phase2_step14/step14_convergence_analysis.csv")
    assert os.path.exists("docs/PHASE2_STEP14_CONFIGURATION_AUDIT.md")
    assert os.path.exists("docs/PHASE2_STEP14_MULTI_SEED_CONVERGENCE.md")
    
    for s in seeds:
        seed_dir = os.path.join(base_dir, f"seed_{s}")
        assert os.path.exists(seed_dir), f"Missing seed directory: {seed_dir}"
        assert os.path.exists(os.path.join(seed_dir, "run_manifest.json"))
        assert os.path.exists(os.path.join(seed_dir, "config.yaml"))
        assert os.path.exists(os.path.join(seed_dir, "training_metrics.json"))
        assert os.path.exists(os.path.join(seed_dir, "evaluation_metrics.json"))
        assert os.path.exists(os.path.join(seed_dir, "training_curve.csv"))
        assert os.path.exists(os.path.join(seed_dir, "evaluation_results.csv"))
        assert os.path.exists(os.path.join(seed_dir, "checkpoint.pt"))

def test_step14_convergence_quality_and_stability():
    """Verify statistical metrics and convergence criteria from Step 14 execution."""
    df = pd.read_csv("results/phase2_step14/step14_seed_summary.csv")
    assert len(df) == 5, f"Expected 5 seeds, found {len(df)}"
    
    # 1. Zero NaNs / Infs
    assert (df["nan_inf_obs"] == 0).all()
    assert (df["nan_inf_q"] == 0).all()
    assert (df["nan_inf_loss"] == 0).all()
    
    # 2. High Completion Ratio (>= 95% across all seeds)
    assert (df["completion_ratio"] >= 0.95).all()
    assert df["completion_ratio"].mean() >= 0.98
    
    # 3. Delay Stability (CV <= 0.05 across seeds)
    mean_delay = df["mean_delay_s"].mean()
    std_delay = df["mean_delay_s"].std()
    cv_delay = std_delay / mean_delay
    assert cv_delay < 0.05, f"Delay CV {cv_delay} exceeds 0.05 threshold"
    
    # 4. Monotonic Loss Reduction
    assert (df["loss_final"] < df["loss_initial"]).all()

def test_step14_checkpoint_recoverability():
    """Verify all 5 saved checkpoints can be cleanly loaded into fresh DDQNAgents."""
    seeds = [42, 43, 44, 45, 46]
    for s in seeds:
        ckpt_path = f"results/phase2_step14/linear_corridor_DDQN_w20/seed_{s}/checkpoint.pt"
        agent = DDQNAgent(input_dim=114, num_actions=7, hidden_dim=128, device="cpu")
        state_dict = torch.load(ckpt_path)
        agent.online_net.load_state_dict(state_dict)
        assert agent.online_net is not None
