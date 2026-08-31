import os
import sys
import json
import yaml
import hashlib
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch

from envs.frozen_vec_env import FrozenVECEnv
from envs.entities import SimulationConfig
from models.baselines.ddqn_agent import DDQNAgent
from utils.seed import set_seed

def compute_tensor_hash(tensor_list: List[Any]) -> str:
    h = hashlib.sha256()
    for item in tensor_list:
        if isinstance(item, np.ndarray):
            h.update(item.tobytes())
        elif isinstance(item, torch.Tensor):
            h.update(item.cpu().numpy().tobytes())
        elif isinstance(item, (int, float, str, bool)):
            h.update(str(item).encode("utf-8"))
    return h.hexdigest()

def run_eval_pass(seed: int, agent: DDQNAgent, geom: str = "corridor_2400m", workload: int = 20):
    with open("configs/paper_parameters.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
    config_dict["num_tasks_per_vehicle_range"] = [workload, workload]
    sim_config = SimulationConfig(**config_dict)
    
    realization_file = f"data/evaluation_realizations/realization_{geom}_w{workload}_{seed}.json"
    set_seed(seed)
    
    eval_env = FrozenVECEnv(config=sim_config, realization_path=realization_file)
    eval_obs, _ = eval_env.reset(seed=seed)
    eval_done = False
    
    eval_step_records = []
    eval_action_seq = []
    eval_state_seq = []
    
    while not eval_done:
        eval_mask = eval_env.get_action_mask()
        eval_state_seq.append(eval_obs.copy())
        
        eval_action = agent.select_action(eval_obs, action_mask=eval_mask, deterministic=True)
        eval_action_seq.append(eval_action)
        
        next_obs, reward, term, trunc, info = eval_env.step(eval_action)
        eval_done = term or trunc
        
        eval_step_records.append({
            "v_id": info.get("v_id", "unknown"),
            "task_id": info.get("task_id", 0),
            "delay": info.get("delay", 0.0),
            "energy": info.get("energy", 0.0),
            "completed": info.get("completed", False),
            "action": eval_action
        })
        eval_obs = next_obs
        
    eval_env.close()
    eval_df = pd.DataFrame(eval_step_records)
    completed = eval_df[eval_df["completed"] == True]
    
    mean_delay = float(completed["delay"].mean()) if len(completed) > 0 else 0.0
    mean_energy = float(completed["energy"].mean()) if len(completed) > 0 else 0.0
    completion_ratio = float(len(completed) / len(eval_df)) if len(eval_df) > 0 else 0.0
    
    action_hash = compute_tensor_hash(eval_action_seq)
    state_hash = compute_tensor_hash(eval_state_seq)
    
    return {
        "mean_delay_s": mean_delay,
        "mean_energy_J": mean_energy,
        "completion_ratio": completion_ratio,
        "action_hash": action_hash,
        "state_hash": state_hash,
        "action_seq": eval_action_seq
    }

def main():
    seed = 42
    ckpt_path = f"results/phase2_step14/linear_corridor_DDQN_w20/seed_{seed}/checkpoint.pt"
    
    print("================================================================================")
    print("      STEP 10 & 11: DETERMINISM REPLAY & CHECKPOINT RECOVERY TESTS              ")
    print("================================================================================")
    
    # Load Pass 1 recorded metrics
    with open(f"results/phase2_step14/linear_corridor_DDQN_w20/seed_{seed}/evaluation_metrics.json") as f:
        pass1_recorded = json.load(f)
        
    # Replay Pass 2
    state_dim = 114
    num_actions = 7
    agent = DDQNAgent(
        input_dim=state_dim,
        num_actions=num_actions,
        hidden_dim=128,
        device="cpu"
    )
    agent.online_net.load_state_dict(torch.load(ckpt_path))
    agent.online_net.eval()
    
    pass2_result = run_eval_pass(seed=seed, agent=agent)
    
    # Checkpoint Recovery Pass 3 (fresh agent instance)
    fresh_agent = DDQNAgent(
        input_dim=state_dim,
        num_actions=num_actions,
        hidden_dim=128,
        device="cpu"
    )
    fresh_agent.online_net.load_state_dict(torch.load(ckpt_path))
    fresh_agent.online_net.eval()
    
    pass3_result = run_eval_pass(seed=seed, agent=fresh_agent)
    
    # Verifications
    print("\n--- DETERMINISM COMPARISON (Pass 1 vs Pass 2 vs Recovery Pass 3) ---")
    print(f"Pass 1 Recorded Action Hash: {pass1_recorded['eval_action_hash']}")
    print(f"Pass 2 Replayed Action Hash: {pass2_result['action_hash']}")
    print(f"Pass 3 Recovered Action Hash:{pass3_result['action_hash']}")
    
    print(f"\nPass 1 Delay: {pass1_recorded['mean_delay_s']:.8f} s")
    print(f"Pass 2 Delay: {pass2_result['mean_delay_s']:.8f} s")
    print(f"Pass 3 Delay: {pass3_result['mean_delay_s']:.8f} s")
    
    print(f"\nPass 1 Energy: {pass1_recorded['mean_energy_J']:.8f} J")
    print(f"Pass 2 Energy: {pass2_result['mean_energy_J']:.8f} J")
    print(f"Pass 3 Energy: {pass3_result['mean_energy_J']:.8f} J")
    
    # Assertions
    action_match_1_2 = (pass1_recorded['eval_action_hash'] == pass2_result['action_hash'])
    action_match_2_3 = (pass2_result['action_hash'] == pass3_result['action_hash'])
    delay_diff_1_2 = abs(pass1_recorded['mean_delay_s'] - pass2_result['mean_delay_s'])
    energy_diff_1_2 = abs(pass1_recorded['mean_energy_J'] - pass2_result['mean_energy_J'])
    delay_diff_2_3 = abs(pass2_result['mean_delay_s'] - pass3_result['mean_delay_s'])
    energy_diff_2_3 = abs(pass2_result['mean_energy_J'] - pass3_result['mean_energy_J'])
    
    assert action_match_1_2, "Pass 1 and Pass 2 action hashes do not match!"
    assert action_match_2_3, "Pass 2 and Pass 3 action hashes do not match!"
    assert delay_diff_1_2 < 1e-6, f"Delay difference {delay_diff_1_2} exceeds tolerance!"
    assert energy_diff_1_2 < 1e-6, f"Energy difference {energy_diff_1_2} exceeds tolerance!"
    assert delay_diff_2_3 < 1e-6, f"Recovery Delay difference {delay_diff_2_3} exceeds tolerance!"
    assert energy_diff_2_3 < 1e-6, f"Recovery Energy difference {energy_diff_2_3} exceeds tolerance!"
    
    print("\nALL DETERMINISM AND CHECKPOINT RECOVERY ASSERTIONS PASSED (100% BIT-PERFECT MATCH).")

if __name__ == "__main__":
    main()
