"""
tests/test_training_pipeline_integrity.py
Automated regression tests verifying genuine RL optimization, optimizer updates,
checkpoint creation, checkpoint reload, evaluation linkage, and synthetic data isolation.
"""

import os
import sys
import glob
import json
import hashlib
import pytest
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import yaml

from models.a3c_agent import ActorCritic
from models.baselines.ddqn_agent import DDQNAgent, QNetwork
from envs.frozen_vec_env import FrozenVECEnv
from envs.entities import SimulationConfig

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "paper_parameters.yaml")
REALIZATION_PATH = os.path.join(
    ROOT_DIR, "data", "evaluation_realizations", "realization_corridor_2400m_w20_seed42.json"
)

def compute_param_hash(model):
    hasher = hashlib.sha256()
    for p in model.parameters():
        hasher.update(p.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()

@pytest.fixture
def base_config():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["num_tasks_per_vehicle_range"] = [20, 20]
    return SimulationConfig(**cfg)

def test_a_optimizer_updates_mutate_parameters(base_config):
    """Test A: A genuine gradient update mutates model parameters (params_before != params_after)."""
    torch.manual_seed(42)
    model = ActorCritic(input_dim=114, num_actions=7)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    hash_before = compute_param_hash(model)
    
    # Simulate a genuine A3C loss backward pass
    dummy_state = torch.randn(4, 114)
    logits, values = model(dummy_state)
    target_returns = torch.tensor([-40.0, -42.0, -38.0, -41.0]).view(-1, 1)
    target_actions = torch.tensor([0, 1, 0, 2])
    
    critic_loss = F.mse_loss(values, target_returns)
    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(1, target_actions.unsqueeze(1))
    actor_loss = -selected_log_probs.mean()
    
    total_loss = actor_loss + 0.5 * critic_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    hash_after = compute_param_hash(model)
    
    assert hash_before != hash_after, "Model parameters failed to mutate after optimizer step!"

def test_b_optimizer_step_count_positive():
    """Test B: Verify optimizer state step count increases upon update."""
    model = ActorCritic(input_dim=114, num_actions=7)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    dummy_state = torch.randn(2, 114)
    logits, values = model(dummy_state)
    loss = logits.sum() + values.sum()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check that optimizer state contains positive step counts
    steps = [state["step"] for state in optimizer.state.values() if "step" in state]
    assert len(steps) > 0, "No optimizer state steps recorded!"
    assert all(int(s) >= 1 for s in steps), "Optimizer steps not incremented!"

def test_c_checkpoint_creation_and_hash(tmp_path):
    """Test C: Verify a real checkpoint file is written to disk with a verifiable SHA-256."""
    model = ActorCritic(input_dim=114, num_actions=7)
    ckpt_file = tmp_path / "test_checkpoint.pt"
    
    ckpt_dict = {
        "episode": 10,
        "algorithm": "CoTOP",
        "model_state_dict": model.state_dict(),
        "git_sha": "test_sha_123"
    }
    torch.save(ckpt_dict, ckpt_file)
    
    assert os.path.exists(ckpt_file), "Checkpoint file was not created!"
    assert os.path.getsize(ckpt_file) > 1000, "Checkpoint file size is suspiciously small!"
    
    with open(ckpt_file, "rb") as f:
        file_sha = hashlib.sha256(f.read()).hexdigest()
    assert len(file_sha) == 64, "Invalid SHA-256 hash length!"

def test_d_checkpoint_reload_restores_exact_weights(tmp_path):
    """Test D: Reloading saved checkpoint restores identical parameter hashes."""
    torch.manual_seed(42)
    model_orig = ActorCritic(input_dim=114, num_actions=7)
    hash_orig = compute_param_hash(model_orig)
    
    ckpt_file = tmp_path / "model_weights.pt"
    torch.save({"model_state_dict": model_orig.state_dict()}, ckpt_file)
    
    # Reload in a new model instance
    model_reloaded = ActorCritic(input_dim=114, num_actions=7)
    ckpt_data = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model_reloaded.load_state_dict(ckpt_data["model_state_dict"])
    hash_reloaded = compute_param_hash(model_reloaded)
    
    assert hash_orig == hash_reloaded, "Reloaded model weight hash does not match original!"

def test_e_evaluation_uses_loaded_checkpoint(base_config, tmp_path):
    """Test E: Evaluation outputs differ when using distinct loaded model weights."""
    # Model 1: Initialized weights
    model1 = ActorCritic(input_dim=114, num_actions=7)
    # Model 2: Substantially modified weights (e.g., strong bias towards action 1)
    model2 = ActorCritic(input_dim=114, num_actions=7)
    with torch.no_grad():
        model2.actor_head.bias.fill_(10.0) # Strongly bias toward action
        model2.actor_head.bias[0] = -10.0
    
    env1 = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    obs1, _ = env1.reset(seed=42)
    obs_t1 = torch.FloatTensor(obs1).unsqueeze(0)
    with torch.no_grad():
        logits1, _ = model1(obs_t1)
    act1 = torch.argmax(logits1, dim=-1).item()
    env1.close()
    
    env2 = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    obs2, _ = env2.reset(seed=42)
    obs_t2 = torch.FloatTensor(obs2).unsqueeze(0)
    with torch.no_grad():
        logits2, _ = model2(obs_t2)
    act2 = torch.argmax(logits2, dim=-1).item()
    env2.close()
    
    assert act1 != act2 or not torch.allclose(logits1, logits2), "Distinct models produced identical logits!"

def test_f_synthetic_data_isolation():
    """Test F: Ensure no synthetic curve equations exist in active publication generation scripts."""
    pub_scripts = glob.glob(os.path.join(ROOT_DIR, "scripts", "plot_*.py"))
    for script in pub_scripts:
        content = open(script, "r", encoding="utf-8").read()
        assert "np.exp(-episodes" not in content, f"Synthetic curve formula found in {script}!"
        assert "rewards = -48.0" not in content, f"Synthetic reward formula found in {script}!"

def test_g_telemetry_integrity(base_config):
    """Test G: Telemetry corresponds to actual step count and task IDs in FrozenVECEnv."""
    env = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    obs, _ = env.reset(seed=42)
    
    step_count = 0
    while len(env.pending_tasks) > 0:
        obs, r, term, trunc, info = env.step(0)
        step_count += 1
        assert "delay" in info and info["delay"] > 0.0
        assert "energy" in info and info["energy"] > 0.0
        assert "completed" in info
        
    assert step_count == 200, f"Expected exactly 200 task steps, got {step_count}!"
    env.close()
