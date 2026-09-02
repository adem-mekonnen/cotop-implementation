#!/usr/bin/env python3
"""
scripts/verify_checkpoint_reload.py
Verifies checkpoint reload integrity, parameter state hashes, and evaluation determinism.
"""

import os
import sys
import json
import hashlib
import yaml
import torch
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from envs.frozen_vec_env import FrozenVECEnv
from envs.entities import SimulationConfig
from models.a3c_agent import ActorCritic

def get_state_dict_hash(state_dict):
    hasher = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        hasher.update(k.encode())
        hasher.update(state_dict[k].cpu().numpy().tobytes())
    return hasher.hexdigest()

def main():
    print("=" * 75)
    print("   PHASE 4 — CHECKPOINT RELOAD & EVALUATION VERIFICATION")
    print("=" * 75)

    ckpt_path = os.path.join(
        root_dir,
        "results", "remediation", "training_pipeline_audit", "smoke_test",
        "CoTOP", "corridor_2400m", "w20", "seed_42", "checkpoint.pt"
    )
    
    assert os.path.exists(ckpt_path), f"Checkpoint does not exist: {ckpt_path}"
    
    with open(ckpt_path, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()
        
    print(f"Checkpoint File:   {ckpt_path}")
    print(f"File Size:         {os.path.getsize(ckpt_path)} bytes")
    print(f"File SHA-256:      {ckpt_sha256}")
    
    # Load checkpoint
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint Keys:   {list(ckpt_data.keys())}")
    
    model_state = ckpt_data["model_state_dict"]
    weight_hash = get_state_dict_hash(model_state)
    print(f"Model Weight Hash: {weight_hash}")
    
    # Initialize fresh ActorCritic model
    model = ActorCritic(input_dim=114, num_actions=7)
    model.load_state_dict(model_state)
    model.eval()
    
    # Evaluate deterministically on frozen realization
    with open(os.path.join(root_dir, "configs/paper_parameters.yaml"), "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict["num_tasks_per_vehicle_range"] = [20, 20]
    config = SimulationConfig(**cfg_dict)
    
    realization_path = os.path.join(
        root_dir, "data", "evaluation_realizations", "realization_corridor_2400m_w20_seed42.json"
    )
    env = FrozenVECEnv(config=config, realization_path=realization_path)
    obs, _ = env.reset(seed=42)
    
    action_seq = []
    delays = []
    energies = []
    completed_count = 0
    total_count = 0
    
    while len(env.pending_tasks) > 0:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(obs_t)
        mask = env.get_action_mask()
        mask_t = torch.BoolTensor(mask).unsqueeze(0)
        logits[~mask_t] = -1e9
        action = torch.argmax(logits, dim=-1).item()
        
        action_seq.append(int(action))
        obs, r, term, trunc, info = env.step(action)
        delays.append(info["delay"])
        energies.append(info["energy"])
        if info["completed"]:
            completed_count += 1
        total_count += 1
        
    env.close()
    
    action_sha256 = hashlib.sha256(json.dumps(action_seq).encode()).hexdigest()
    
    mean_delay = float(np.mean(delays))
    mean_energy = float(np.mean(energies))
    comp_ratio = completed_count / total_count
    
    print(f"\n--- RELOAD EVALUATION RESULTS ---")
    print(f"Total Tasks:       {total_count}")
    print(f"Completed Tasks:   {completed_count} ({comp_ratio*100:.2f}%)")
    print(f"Mean Delay:        {mean_delay:.4f} s")
    print(f"Mean Energy:       {mean_energy:.4f} J")
    print(f"Action SHA-256:    {action_sha256}")
    
    # Compare with evaluation_metrics.json from smoke test
    eval_metrics_path = os.path.join(
        root_dir,
        "results", "remediation", "training_pipeline_audit", "smoke_test",
        "CoTOP", "corridor_2400m", "w20", "seed_42", "evaluation_metrics.json"
    )
    with open(eval_metrics_path, "r") as f:
        saved_metrics = json.load(f)
        
    assert abs(mean_delay - saved_metrics["mean_delay_s"]) < 1e-6, "Delay mismatch on reload!"
    assert abs(mean_energy - saved_metrics["mean_energy_j"]) < 1e-6, "Energy mismatch on reload!"
    assert action_sha256 == saved_metrics["action_sequence_sha256"], "Action sequence mismatch on reload!"
    assert ckpt_sha256 == saved_metrics["checkpoint_sha256"], "Checkpoint SHA mismatch!"
    
    print("\n[VERDICT] CHECKPOINT RELOAD & DETERMINISTIC EVALUATION EXACT MATCH: PASS!")
    
    # Save reload test artifact
    out_dir = os.path.join(root_dir, "results", "remediation", "training_pipeline_audit")
    os.makedirs(out_dir, exist_ok=True)
    reload_record = {
        "checkpoint_path": ckpt_path,
        "checkpoint_sha256": ckpt_sha256,
        "weight_hash": weight_hash,
        "reloaded_mean_delay_s": mean_delay,
        "reloaded_mean_energy_j": mean_energy,
        "reloaded_completion_ratio": comp_ratio,
        "action_sequence_sha256": action_sha256,
        "exact_match_verified": True
    }
    with open(os.path.join(out_dir, "reload_test.json"), "w") as f:
        json.dump(reload_record, f, indent=2)
    print(f"Saved reload verification record to {os.path.join(out_dir, 'reload_test.json')}")

if __name__ == "__main__":
    main()
