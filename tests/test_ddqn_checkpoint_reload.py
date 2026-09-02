"""
tests/test_ddqn_checkpoint_reload.py
Automated regression tests verifying DDQN checkpoint serialization, strict loading,
rejection of invalid/missing checkpoints, and deterministic frozen realization replay.
"""

import os
import json
import hashlib
import pytest
import torch
import numpy as np
import yaml

from models.baselines.ddqn_agent import QNetwork, DDQNAgent
from envs.frozen_vec_env import FrozenVECEnv
from envs.entities import SimulationConfig
from utils.checkpoint_io import load_checkpoint_strict, compute_file_sha256, compute_model_param_hash

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

def test_a_ddqn_checkpoint_save_and_strict_reload(tmp_path):
    """Test A: Valid DDQN checkpoint is strictly loaded and restores exact parameters."""
    torch.manual_seed(42)
    model_orig = QNetwork(input_dim=114, num_actions=7)
    param_hash_orig = compute_model_param_hash(model_orig)
    
    ckpt_file = str(tmp_path / "ddqn_test.pt")
    ckpt_dict = {
        "algorithm": "DDQN",
        "online_net_state_dict": model_orig.state_dict(),
        "seed": 42
    }
    torch.save(ckpt_dict, ckpt_file)
    
    model_new = QNetwork(input_dim=114, num_actions=7)
    metadata = load_checkpoint_strict(
        checkpoint_path=ckpt_file,
        model=model_new,
        expected_algorithm="DDQN",
        device="cpu"
    )
    
    param_hash_new = compute_model_param_hash(model_new)
    assert param_hash_orig == param_hash_new, "Restored DDQN parameter hash does not match original!"
    assert metadata["checkpoint_sha256"] == compute_file_sha256(ckpt_file)
    assert metadata["saved_metadata"]["algorithm"] == "DDQN"

def test_b_ddqn_missing_checkpoint_raises_hard_error():
    """Test B: Missing checkpoint path raises FileNotFoundError (no silent fallback)."""
    model = QNetwork(input_dim=114, num_actions=7)
    with pytest.raises(FileNotFoundError) as exc_info:
        load_checkpoint_strict(
            checkpoint_path="non_existent_ddqn_path_12345.pt",
            model=model,
            expected_algorithm="DDQN"
        )
    assert "FATAL ERROR" in str(exc_info.value)
    assert "not found" in str(exc_info.value)

def test_c_ddqn_corrupt_checkpoint_raises_hard_error(tmp_path):
    """Test C: Corrupted or incompatible checkpoint raises RuntimeError."""
    corrupt_file = str(tmp_path / "corrupt_ddqn.pt")
    with open(corrupt_file, "w") as f:
        f.write("corrupted non-pytorch payload")
        
    model = QNetwork(input_dim=114, num_actions=7)
    with pytest.raises(RuntimeError) as exc_info:
        load_checkpoint_strict(
            checkpoint_path=corrupt_file,
            model=model,
            expected_algorithm="DDQN"
        )
    assert "FATAL ERROR" in str(exc_info.value)

def test_d_ddqn_algorithm_mismatch_raises_hard_error(tmp_path):
    """Test D: Mismatched algorithm key raises ValueError."""
    ckpt_file = str(tmp_path / "mismatch_algo.pt")
    model = QNetwork(input_dim=114, num_actions=7)
    torch.save({
        "algorithm": "CoTOP", # Wrong algorithm for DDQN
        "model_state_dict": model.state_dict()
    }, ckpt_file)
    
    with pytest.raises(ValueError) as exc_info:
        load_checkpoint_strict(
            checkpoint_path=ckpt_file,
            model=model,
            expected_algorithm="DDQN"
        )
    assert "algorithm mismatch" in str(exc_info.value)

def test_e_ddqn_deterministic_frozen_evaluation(base_config, tmp_path):
    """Test E: DDQN model evaluated on frozen realization produces deterministic bitwise outputs."""
    torch.manual_seed(42)
    model = QNetwork(input_dim=114, num_actions=7)
    ckpt_file = str(tmp_path / "ddqn_eval.pt")
    torch.save({"online_net_state_dict": model.state_dict(), "algorithm": "DDQN"}, ckpt_file)
    
    # Eval 1
    m1 = QNetwork(input_dim=114, num_actions=7)
    load_checkpoint_strict(ckpt_file, m1, expected_algorithm="DDQN")
    env1 = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    obs1, _ = env1.reset(seed=42)
    actions1, delays1 = [], []
    while len(env1.pending_tasks) > 0:
        mask = env1.get_action_mask()
        obs_t = torch.FloatTensor(obs1).unsqueeze(0)
        with torch.no_grad():
            logits = m1(obs_t)
        mask_t = torch.BoolTensor(mask).unsqueeze(0)
        logits[~mask_t] = -1e9
        act = torch.argmax(logits, dim=-1).item()
        actions1.append(act)
        obs1, _, _, _, info = env1.step(act)
        delays1.append(info["delay"])
    env1.close()
    
    # Eval 2
    m2 = QNetwork(input_dim=114, num_actions=7)
    load_checkpoint_strict(ckpt_file, m2, expected_algorithm="DDQN")
    env2 = FrozenVECEnv(config=base_config, realization_path=REALIZATION_PATH)
    obs2, _ = env2.reset(seed=42)
    actions2, delays2 = [], []
    while len(env2.pending_tasks) > 0:
        mask = env2.get_action_mask()
        obs_t = torch.FloatTensor(obs2).unsqueeze(0)
        with torch.no_grad():
            logits = m2(obs_t)
        mask_t = torch.BoolTensor(mask).unsqueeze(0)
        logits[~mask_t] = -1e9
        act = torch.argmax(logits, dim=-1).item()
        actions2.append(act)
        obs2, _, _, _, info = env2.step(act)
        delays2.append(info["delay"])
    env2.close()
    
    assert actions1 == actions2, "DDQN action sequences differed across evaluations!"
    assert np.allclose(delays1, delays2), "DDQN delays differed across evaluations!"
    assert hashlib.sha256(json.dumps(actions1).encode()).hexdigest() == hashlib.sha256(json.dumps(actions2).encode()).hexdigest()
