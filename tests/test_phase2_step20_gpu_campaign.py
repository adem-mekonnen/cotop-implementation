import pytest
import os
import sys
import json
import hashlib
import tempfile
import torch
import numpy as np
import pandas as pd

from scripts.run_phase2_gpu_campaign import (
    verify_physics_hashes,
    get_hardware_info,
    capture_rng_state,
    restore_rng_state,
    load_sim_config,
    ensure_realization,
    run_training_and_eval,
    COMM_SHA256,
    COMP_SHA256
)
from models.baselines.ddqn_agent import DDQNAgent, QNetwork
from models.a3c_agent import ActorCritic
from utils.seed import set_seed

# 1. Physics Hashes Invariant Test
def test_01_physics_hashes_integrity():
    comm_h, comp_h = verify_physics_hashes()
    assert comm_h == COMM_SHA256, "comm_model.py hash altered!"
    assert comp_h == COMP_SHA256, "comp_model.py hash altered!"

# 2. Loud Failure on Missing CUDA without --allow-cpu
def test_02_cuda_loud_failure_when_unavailable():
    if not torch.cuda.is_available():
        with pytest.raises(SystemExit):
            get_hardware_info("cuda:0", allow_cpu=False)

# 3. Hardware Info Structure
def test_03_hardware_info_structure():
    info = get_hardware_info("cpu", allow_cpu=True)
    assert "device" in info
    assert "cuda_available" in info
    assert "gpu_name" in info
    assert "pytorch_ver" in info
    assert "python_ver" in info

# 4. Checkpoint Content & Recovery
def test_04_checkpoint_content_and_recovery():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
        model = QNetwork(input_dim=114, num_actions=7)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
        
        # Save comprehensive checkpoint
        torch.save({
            "episode": 10,
            "global_step": 2000,
            "epsilon": 0.5,
            "online_net_state_dict": model.state_dict(),
            "target_net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "git_sha": "test_sha",
            "physics_hashes": {"comm": COMM_SHA256, "comp": COMP_SHA256},
            "rng_state": capture_rng_state(torch.device("cpu"))
        }, ckpt_path)
        
        # Load and verify
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert loaded["episode"] == 10
        assert loaded["global_step"] == 2000
        assert loaded["epsilon"] == 0.5
        assert loaded["physics_hashes"]["comm"] == COMM_SHA256
        assert loaded["physics_hashes"]["comp"] == COMP_SHA256
        assert "rng_state" in loaded

# 5. Output Isolation Structure
def test_05_output_isolation_structure():
    with tempfile.TemporaryDirectory() as tmpdir:
        dev_info = get_hardware_info("cpu", allow_cpu=True)
        # Execute 1-episode smoke run
        run_training_and_eval(
            algorithm="DDQN",
            scenario="corridor_2400m",
            workload=20,
            seed=42,
            episodes=1,
            device_info=dev_info,
            output_base_dir=tmpdir,
            resume=False,
            checkpoint_interval=1
        )
        
        expected_dir = os.path.join(tmpdir, "DDQN", "corridor_2400m", "w20", "seed_42")
        assert os.path.exists(expected_dir), f"Directory {expected_dir} not created!"
        assert os.path.exists(os.path.join(expected_dir, "checkpoint.pt"))
        assert os.path.exists(os.path.join(expected_dir, "config.yaml"))
        assert os.path.exists(os.path.join(expected_dir, "run_manifest.json"))
        assert os.path.exists(os.path.join(expected_dir, "realization_manifest.json"))
        assert os.path.exists(os.path.join(expected_dir, "training_metrics.json"))
        assert os.path.exists(os.path.join(expected_dir, "evaluation_metrics.json"))
        assert os.path.exists(os.path.join(expected_dir, "training_curve.csv"))
        assert os.path.exists(os.path.join(expected_dir, "evaluation_results.csv"))

# 6. Resume Mechanism Protection
def test_06_resume_mechanism():
    with tempfile.TemporaryDirectory() as tmpdir:
        dev_info = get_hardware_info("cpu", allow_cpu=True)
        # 1. Run 1 episode
        run_training_and_eval(
            algorithm="DDQN",
            scenario="corridor_2400m",
            workload=20,
            seed=42,
            episodes=1,
            device_info=dev_info,
            output_base_dir=tmpdir,
            resume=False,
            checkpoint_interval=1
        )
        
        eval_path = os.path.join(tmpdir, "DDQN", "corridor_2400m", "w20", "seed_42", "evaluation_metrics.json")
        mtime_first = os.path.getmtime(eval_path)
        
        # 2. Run with resume=True (should detect completion and skip)
        run_training_and_eval(
            algorithm="DDQN",
            scenario="corridor_2400m",
            workload=20,
            seed=42,
            episodes=1,
            device_info=dev_info,
            output_base_dir=tmpdir,
            resume=True,
            checkpoint_interval=1
        )
        mtime_second = os.path.getmtime(eval_path)
        assert mtime_first == mtime_second, "Completed run was re-executed instead of skipped on resume!"

# 7. Experiment Manifest Schema
def test_07_experiment_manifest_schema():
    manifest_path = "results/phase2_step20/DDQN/corridor_2400m/w20/seed_42/run_manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        assert "git_commit_sha" in manifest
        assert "timestamp" in manifest
        assert "algorithm" in manifest
        assert "scenario" in manifest
        assert "workload" in manifest
        assert "seed" in manifest
        assert "hardware" in manifest
        assert "software" in manifest
        assert "physics_hashes" in manifest
        assert "realization_sha256" in manifest
        assert "checkpoint_sha256" in manifest
        assert manifest["physics_hashes"]["comm_model_sha256"] == COMM_SHA256
        assert manifest["physics_hashes"]["comp_model_sha256"] == COMP_SHA256
