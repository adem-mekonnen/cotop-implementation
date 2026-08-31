import os
import sys
import json
import yaml
import time
import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch

from envs.frozen_vec_env import FrozenVECEnv
from envs.entities import SimulationConfig
from models.baselines.ddqn_agent import DDQNAgent
from utils.seed import set_seed

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

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

def train_and_evaluate_seed(seed: int, num_episodes: int = 500, base_out_dir: str = "results/phase2_step14"):
    geom = "corridor_2400m"
    workload = 20
    out_dir = os.path.join(base_out_dir, f"linear_corridor_DDQN_w20/seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    
    eval_metrics_file = os.path.join(out_dir, "evaluation_metrics.json")
    if os.path.exists(eval_metrics_file):
        print(f"Seed {seed} already completed. Loading existing metrics...")
        with open(os.path.join(out_dir, "training_metrics.json")) as f:
            tm = json.load(f)
        with open(eval_metrics_file) as f:
            em = json.load(f)
        with open(os.path.join(out_dir, "run_manifest.json")) as f:
            rm = json.load(f)
        return {
            "seed": seed,
            **tm,
            **em,
            "model_hash": rm["model_hash"],
            "realization_hash": rm["realization_hash"]
        }
    
    # 1. Environment and Config
    with open("configs/paper_parameters.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
    config_dict["num_tasks_per_vehicle_range"] = [workload, workload]
    sim_config = SimulationConfig(**config_dict)
    
    realization_file = f"data/evaluation_realizations/realization_{geom}_w{workload}_{seed}.json"
    realization_hash = compute_sha256(realization_file)
    
    set_seed(seed)
    
    # 2. Agent Initialization
    train_env = FrozenVECEnv(config=sim_config, realization_path=realization_file)
    state_dim = train_env.observation_space.shape[0]
    num_actions = train_env.action_space.n
    
    agent = DDQNAgent(
        input_dim=state_dim,
        num_actions=num_actions,
        hidden_dim=128,
        learning_rate=0.0002,
        gamma=0.99,
        replay_capacity=10000,
        batch_size=64,
        target_update_frequency=100,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=200,
        device="cpu"
    )
    
    training_curve = []
    opt_steps = 0
    target_sync_count = 0
    nan_inf_obs = 0
    nan_inf_q = 0
    nan_inf_loss = 0
    
    print(f"Starting DDQN Training: Seed {seed} for {num_episodes} episodes...")
    start_time = time.time()
    
    for ep in range(num_episodes):
        obs, _ = train_env.reset()
        done = False
        ep_reward = 0.0
        ep_losses = []
        ep_q_values = []
        
        while not done:
            mask = train_env.get_action_mask()
            if np.isnan(obs).any() or np.isinf(obs).any():
                nan_inf_obs += 1
                
            # Compute Q-values for NaN check
            with torch.no_grad():
                q_vals = agent.online_net(torch.FloatTensor(obs).unsqueeze(0).to(agent.device)).squeeze(0).cpu().numpy()
                if np.isnan(q_vals).any() or np.isinf(q_vals).any():
                    nan_inf_q += 1
                ep_q_values.append(float(np.mean(q_vals)))
                
            action = agent.select_action(obs, action_mask=mask, deterministic=False)
            next_obs, reward, term, trunc, info = train_env.step(action)
            done = term or trunc
            
            next_mask = train_env.get_action_mask() if not done else None
            agent.store_transition(obs, action, reward, next_obs, done, next_action_mask=next_mask)
            
            loss = agent.update()
            if loss is not None:
                if np.isnan(loss) or np.isinf(loss):
                    nan_inf_loss += 1
                ep_losses.append(loss)
                opt_steps += 1
                if opt_steps % agent.target_update_frequency == 0:
                    target_sync_count += 1
                    
            obs = next_obs
            ep_reward += reward
            
        agent.set_episode(ep + 1)
        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        mean_q = float(np.mean(ep_q_values)) if ep_q_values else 0.0
        
        training_curve.append({
            "episode": ep + 1,
            "reward": float(ep_reward),
            "loss": mean_loss,
            "mean_q": mean_q,
            "epsilon": float(agent.epsilon),
            "buffer_size": len(agent.memory)
        })
        
    train_duration = time.time() - start_time
    train_env.close()
    
    # Save Checkpoint
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    torch.save(agent.online_net.state_dict(), ckpt_path)
    model_hash = compute_sha256(ckpt_path)
    
    # 3. Deterministic Evaluation
    eval_env = FrozenVECEnv(config=sim_config, realization_path=realization_file)
    eval_obs, _ = eval_env.reset(seed=seed)
    eval_done = False
    
    eval_step_records = []
    eval_action_seq = []
    eval_state_seq = []
    
    while not eval_done:
        eval_mask = eval_env.get_action_mask()
        eval_state_seq.append(eval_obs.copy())
        
        # Deterministic greedy action selection with epsilon = 0.0
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
            "comm_delay": info.get("comm_delay", 0.0),
            "comp_delay": info.get("comp_delay", 0.0),
            "wait_delay": info.get("wait_delay", 0.0),
            "action": eval_action
        })
        eval_obs = next_obs
        
    eval_env.close()
    
    eval_df = pd.DataFrame(eval_step_records)
    completed_tasks = eval_df[eval_df["completed"] == True]
    failed_tasks = eval_df[eval_df["completed"] == False]
    
    mean_delay = float(completed_tasks["delay"].mean()) if len(completed_tasks) > 0 else 0.0
    median_delay = float(completed_tasks["delay"].median()) if len(completed_tasks) > 0 else 0.0
    std_delay = float(completed_tasks["delay"].std(ddof=1)) if len(completed_tasks) > 1 else 0.0
    min_delay = float(completed_tasks["delay"].min()) if len(completed_tasks) > 0 else 0.0
    max_delay = float(completed_tasks["delay"].max()) if len(completed_tasks) > 0 else 0.0
    
    mean_energy = float(completed_tasks["energy"].mean()) if len(completed_tasks) > 0 else 0.0
    median_energy = float(completed_tasks["energy"].median()) if len(completed_tasks) > 0 else 0.0
    std_energy = float(completed_tasks["energy"].std(ddof=1)) if len(completed_tasks) > 1 else 0.0
    min_energy = float(completed_tasks["energy"].min()) if len(completed_tasks) > 0 else 0.0
    max_energy = float(completed_tasks["energy"].max()) if len(completed_tasks) > 0 else 0.0
    
    completion_ratio = float(len(completed_tasks) / len(eval_df)) if len(eval_df) > 0 else 0.0
    
    eval_action_hash = compute_tensor_hash(eval_action_seq)
    eval_state_hash = compute_tensor_hash(eval_state_seq)
    
    # Calculate Training Reward Diagnostics
    rewards = [r["reward"] for r in training_curve]
    first_50_rewards = rewards[:50]
    last_50_rewards = rewards[-50:]
    losses = [r["loss"] for r in training_curve if r["loss"] > 0]
    
    train_metrics = {
        "num_episodes": num_episodes,
        "train_duration_s": train_duration,
        "optimization_steps": opt_steps,
        "target_sync_count": target_sync_count,
        "reward_first_50_mean": float(np.mean(first_50_rewards)),
        "reward_first_50_std": float(np.std(first_50_rewards, ddof=1)),
        "reward_last_50_mean": float(np.mean(last_50_rewards)),
        "reward_last_50_std": float(np.std(last_50_rewards, ddof=1)),
        "reward_overall_mean": float(np.mean(rewards)),
        "reward_overall_std": float(np.std(rewards, ddof=1)),
        "loss_initial": float(losses[0]) if losses else 0.0,
        "loss_final": float(losses[-1]) if losses else 0.0,
        "loss_mean": float(np.mean(losses)) if losses else 0.0,
        "loss_std": float(np.std(losses, ddof=1)) if losses else 0.0,
        "final_epsilon": float(agent.epsilon),
        "final_buffer_size": len(agent.memory),
        "nan_inf_obs": nan_inf_obs,
        "nan_inf_q": nan_inf_q,
        "nan_inf_loss": nan_inf_loss
    }
    
    eval_metrics = {
        "tasks_generated": len(eval_df),
        "tasks_completed": len(completed_tasks),
        "tasks_failed": len(failed_tasks),
        "tasks_pending": 0,
        "completion_ratio": completion_ratio,
        "mean_delay_s": mean_delay,
        "median_delay_s": median_delay,
        "std_delay_s": std_delay,
        "min_delay_s": min_delay,
        "max_delay_s": max_delay,
        "mean_energy_J": mean_energy,
        "median_energy_J": median_energy,
        "std_energy_J": std_energy,
        "min_energy_J": min_energy,
        "max_energy_J": max_energy,
        "comm_delay_mean_s": float(completed_tasks["comm_delay"].mean()) if len(completed_tasks) > 0 else 0.0,
        "comp_delay_mean_s": float(completed_tasks["comp_delay"].mean()) if len(completed_tasks) > 0 else 0.0,
        "wait_delay_mean_s": float(completed_tasks["wait_delay"].mean()) if len(completed_tasks) > 0 else 0.0,
        "eval_action_hash": eval_action_hash,
        "eval_state_hash": eval_state_hash,
        "eval_realization_hash": realization_hash
    }
    
    # Save Artifacts
    with open(os.path.join(out_dir, "training_metrics.json"), "w") as f:
        json.dump(train_metrics, f, indent=2)
    with open(os.path.join(out_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(config_dict, f)
        
    pd.DataFrame(training_curve).to_csv(os.path.join(out_dir, "training_curve.csv"), index=False)
    eval_df.to_csv(os.path.join(out_dir, "evaluation_results.csv"), index=False)
    
    # Manifest
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    manifest = {
        "algorithm": "DDQN",
        "geometry": geom,
        "workload": f"w{workload}",
        "seed": seed,
        "git_sha": git_sha,
        "model_hash": model_hash,
        "realization_hash": realization_hash,
        "config_hash": compute_sha256("configs/paper_parameters.yaml"),
        "environment_fingerprint": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "os": f"{platform.system()} {platform.release()}",
            "hardware": platform.processor()
        }
    }
    with open(os.path.join(out_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Completed Seed {seed}: Completion = {completion_ratio*100:.1f}%, Delay = {mean_delay:.4f}s, Energy = {mean_energy:.4f}J")
    return {
        "seed": seed,
        **train_metrics,
        **eval_metrics,
        "model_hash": model_hash,
        "realization_hash": realization_hash
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    
    seeds = args.seeds
    results = []
    
    print("================================================================================")
    print(f"      PHASE 2 — STEP 14: MULTI-SEED DDQN TRAINING (500 EPISODES x {len(seeds)} SEEDS)      ")
    print("================================================================================")
    
    for s in seeds:
        res = train_and_evaluate_seed(s, num_episodes=500)
        results.append(res)
        
    summary_df = pd.DataFrame(results)
    os.makedirs("results/phase2_step14", exist_ok=True)
    summary_df.to_csv("results/phase2_step14/step14_seed_summary.csv", index=False)
    print(f"Saved results/phase2_step14/step14_seed_summary.csv")
    
    # Convergence and Cross-Seed Analysis
    analysis_records = []
    metrics_to_analyze = [
        ("mean_delay_s", "Delay (s)"),
        ("mean_energy_J", "Energy (J)"),
        ("completion_ratio", "Completion Ratio"),
        ("reward_last_50_mean", "Last 50 Reward"),
        ("loss_mean", "Mean Loss"),
        ("optimization_steps", "Optimization Steps")
    ]
    
    for col, name in metrics_to_analyze:
        vals = summary_df[col].values
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals, ddof=1))
        cv_v = float(std_v / mean_v) if abs(mean_v) > 1e-12 else 0.0
        median_v = float(np.median(vals))
        iqr_v = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        min_v = float(np.min(vals))
        max_v = float(np.max(vals))
        
        analysis_records.append({
            "metric_name": name,
            "metric_col": col,
            "mean": mean_v,
            "std": std_v,
            "cv": cv_v,
            "median": median_v,
            "iqr": iqr_v,
            "min": min_v,
            "max": max_v,
            "seed_vector": str(list(np.round(vals, 4)))
        })
        
    analysis_df = pd.DataFrame(analysis_records)
    analysis_df.to_csv("results/phase2_step14/step14_convergence_analysis.csv", index=False)
    print(f"Saved results/phase2_step14/step14_convergence_analysis.csv")

if __name__ == "__main__":
    main()
