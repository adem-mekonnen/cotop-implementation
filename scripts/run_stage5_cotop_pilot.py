import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import yaml

from envs.entities import SimulationConfig
from envs.vec_env import VECEnv
from models.a3c_agent import ActorCritic
from utils.seed import set_seed


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compute_param_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for p in model.state_dict().values():
        h.update(p.cpu().numpy().tobytes())
    return h.hexdigest()


def classify_convergence(rewards: List[float], losses: List[float], nan_inf_count: int) -> str:
    if nan_inf_count > 0:
        return "NUMERICALLY_INVALID"
    if len(rewards) < 100:
        return "NON_CONVERGED / INSUFFICIENT_EVIDENCE"
    
    last_100_rewards = rewards[-100:]
    last_50_losses = [l for l in losses[-50:] if l is not None and np.isfinite(l)]
    
    mean_r = np.mean(last_100_rewards)
    std_r = np.std(last_100_rewards)
    cv_r = std_r / (abs(mean_r) + 1e-6)
    
    if last_50_losses and np.mean(last_50_losses) > 500.0:
        return "DIVERGED"
    elif cv_r < 0.15:
        return "STABLE"
    elif cv_r < 0.40:
        return "OSCILLATORY"
    else:
        return "NON_CONVERGED"


def run_cotop_training(geometry: str, seed: int, port: int, episodes: int = 500) -> Dict:
    start_time = time.time()
    
    master_seed = seed
    env_seed = 10000 + seed
    train_seed = 20000 + seed
    eval_seed = 30000 + seed
    
    set_seed(train_seed)
    
    output_dir = f"results/stage5_cotop_retrain/{geometry}/seed_{seed}"
    os.makedirs(output_dir, exist_ok=True)
    
    manifest_path = os.path.join(output_dir, "run_manifest.json")
    ckpt_path = os.path.join(output_dir, "checkpoint_ep500.pt")
    metrics_path = os.path.join(output_dir, "metrics.json")
    curve_path = os.path.join(output_dir, "training_curve.csv")
    results_path = os.path.join(output_dir, "seed_results.csv")
    
    # Load config
    with open("configs/paper_parameters.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict["num_tasks_per_vehicle_range"] = [20, 20]
    config = SimulationConfig(**cfg_dict)
    
    sim_geom = "grid_200m" if geometry in ["grid_200m", "urban_manhattan"] else "corridor_2400m"
    
    # Initialize environment
    env = VECEnv(
        config=config,
        port=port,
        scenario_geometry=sim_geom,
        use_mobility_model=True,
        use_priority=True,
        priority_mode="paper_literal",
        coverage_mode="completion_position",
        spatial_graph_radius=200.0,
        max_vehicles=10,
        seed=env_seed
    )
    
    obs_dim = env.observation_space.shape[0]  # 114 for w=20
    num_actions = env.action_space.n          # 7
    
    model = ActorCritic(input_dim=obs_dim, num_actions=num_actions, hidden_size=128)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    gamma = 0.99
    entropy_coef = 0.01
    
    training_records = []
    ep_rewards = []
    ep_losses = []
    nan_inf_count = 0
    optimization_steps = 0
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=env_seed + ep)
        done = False
        
        values = []
        log_probs = []
        rewards = []
        entropies = []
        
        ep_delay = 0.0
        ep_energy = 0.0
        ep_tasks = 0
        ep_completed = 0
        
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits, val = model(obs_t)
            
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            values.append(val.squeeze(0))
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
            
            ep_tasks += 1
            if "delay" in info:
                ep_delay += info["delay"]
                ep_energy += info["energy"]
                if info.get("completed", False):
                    ep_completed += 1
                    
            obs = next_obs
            
        # Compute discounted returns
        R = 0.0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
            
        returns_t = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.cat(values).squeeze(-1)
        log_probs_t = torch.cat(log_probs)
        entropies_t = torch.cat(entropies)
        
        advantages = returns_t - values_t.detach()
        actor_loss = -(log_probs_t * advantages).mean()
        critic_loss = F.mse_loss(values_t, returns_t)
        total_loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropies_t.mean()
        
        # Gradient update
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=40.0)
        optimizer.step()
        optimization_steps += 1
        
        loss_val = float(total_loss.item())
        if not np.isfinite(loss_val):
            nan_inf_count += 1
            
        total_ep_reward = float(sum(rewards))
        ep_rewards.append(total_ep_reward)
        ep_losses.append(loss_val)
        
        avg_delay = float(ep_delay / max(ep_tasks, 1))
        avg_energy = float(ep_energy / max(ep_tasks, 1))
        comp_ratio = float(ep_completed / max(ep_tasks, 1))
        
        training_records.append({
            "episode": ep + 1,
            "reward": total_ep_reward,
            "loss": loss_val,
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropies_t.mean().item()),
            "avg_delay": avg_delay,
            "avg_energy": avg_energy,
            "completion_ratio": comp_ratio
        })
        
    env.close()
    
    # Save checkpoint at episode 500
    torch.save({
        "episode": episodes,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "input_dim": obs_dim,
        "num_actions": num_actions,
        "seed": seed,
        "geometry": geometry
    }, ckpt_path)
    
    ckpt_hash = compute_file_sha256(ckpt_path)
    
    # Write training_curve.csv
    with open(curve_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(training_records[0].keys()))
        writer.writeheader()
        writer.writerows(training_records)
        
    # Deterministic Evaluation Pass
    eval_env = VECEnv(
        config=config,
        port=port + 100,
        scenario_geometry=sim_geom,
        use_mobility_model=True,
        use_priority=True,
        priority_mode="paper_literal",
        coverage_mode="completion_position",
        spatial_graph_radius=200.0,
        max_vehicles=10,
        seed=eval_seed
    )
    
    model.eval()
    obs, _ = eval_env.reset(seed=eval_seed)
    eval_done = False
    
    eval_delays = []
    eval_energies = []
    eval_completed = 0
    eval_total = 0
    
    comm_delays = []
    comp_delays = []
    wait_delays = []
    
    with torch.no_grad():
        while not eval_done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits, _ = model(obs_t)
            action = torch.argmax(logits, dim=-1).item()
            
            obs, reward, terminated, truncated, info = eval_env.step(action)
            eval_done = terminated or truncated
            
            eval_total += 1
            if "delay" in info:
                eval_delays.append(info["delay"])
                eval_energies.append(info["energy"])
                comm_delays.append(info.get("comm_delay", 0.0))
                comp_delays.append(info.get("comp_delay", 0.0))
                wait_delays.append(info.get("wait_delay", 0.0))
                if info.get("completed", False):
                    eval_completed += 1
                    
    eval_env.close()
    
    wall_time = time.time() - start_time
    conv_status = classify_convergence(ep_rewards, ep_losses, nan_inf_count)
    
    mean_delay = float(np.mean(eval_delays)) if eval_delays else 0.0
    mean_energy = float(np.mean(eval_energies)) if eval_energies else 0.0
    completion_ratio = float(eval_completed / max(eval_total, 1))
    
    # Save seed_results.csv
    seed_result_data = [{
        "algorithm": "CoTOP",
        "geometry": geometry,
        "workload": 20,
        "seed": seed,
        "episodes": episodes,
        "optimization_steps": optimization_steps,
        "wall_clock_time_s": round(wall_time, 2),
        "mean_reward_last100": round(float(np.mean(ep_rewards[-100:])), 4),
        "convergence_status": conv_status,
        "eval_mean_delay": round(mean_delay, 4),
        "eval_mean_energy": round(mean_energy, 4),
        "eval_completion_ratio": round(completion_ratio, 4),
        "eval_comm_delay": round(float(np.mean(comm_delays)), 4) if comm_delays else 0.0,
        "eval_comp_delay": round(float(np.mean(comp_delays)), 4) if comp_delays else 0.0,
        "eval_wait_delay": round(float(np.mean(wait_delays)), 4) if wait_delays else 0.0,
        "checkpoint_sha256": ckpt_hash
    }]
    
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(seed_result_data[0].keys()))
        writer.writeheader()
        writer.writerows(seed_result_data)
        
    # Save metrics.json
    metrics_data = {
        "algorithm": "CoTOP",
        "geometry": geometry,
        "workload": 20,
        "seed": seed,
        "episodes": episodes,
        "optimization_steps": optimization_steps,
        "wall_clock_time_s": wall_time,
        "nan_inf_count": nan_inf_count,
        "convergence_classification": conv_status,
        "mean_training_reward_last100": float(np.mean(ep_rewards[-100:])),
        "std_training_reward_last100": float(np.std(ep_rewards[-100:])),
        "evaluation_metrics": {
            "mean_delay_s": mean_delay,
            "mean_energy_j": mean_energy,
            "completion_ratio": completion_ratio,
            "mean_comm_delay_s": float(np.mean(comm_delays)) if comm_delays else 0.0,
            "mean_comp_delay_s": float(np.mean(comp_delays)) if comp_delays else 0.0,
            "mean_wait_delay_s": float(np.mean(wait_delays)) if wait_delays else 0.0
        }
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
        
    # Save run_manifest.json
    manifest_data = {
        "algorithm": "CoTOP",
        "geometry": geometry,
        "workload": 20,
        "seed": seed,
        "master_seed": master_seed,
        "env_seed": env_seed,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "episodes": episodes,
        "git_commit_sha": "a43abc5",
        "comm_model_sha256": compute_file_sha256("envs/comm_model.py"),
        "comp_model_sha256": compute_file_sha256("envs/comp_model.py"),
        "checkpoint_sha256": ckpt_hash,
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "wall_clock_time_s": wall_time,
        "convergence_status": conv_status
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
        
    print(f"[{geometry} | CoTOP | Seed {seed}] Done in {wall_time:.1f}s | Conv: {conv_status} | Delay: {mean_delay:.2f}s | Energy: {mean_energy:.2f}J | Comp: {completion_ratio*100:.1f}%")
    return seed_result_data[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    
    geometries = ["corridor_2400m", "grid_200m"]
    seeds = [0, 1, 2, 3, 4]
    
    all_results = []
    port = 9300
    
    print(f"============================================================")
    print(f"   STAGE 5: RETRAINING CoTOP BASELINE (CANONICAL PHASE 2)   ")
    print(f"   Geometries: {geometries} | Seeds: {seeds} | Ep: {args.episodes}")
    print(f"============================================================")
    
    for geom in geometries:
        for s in seeds:
            res = run_cotop_training(geometry=geom, seed=s, port=port, episodes=args.episodes)
            all_results.append(res)
            port += 2
            
    summary_path = "results/stage5_cotop_retrain/STAGE5_COTOP_PILOT_SUMMARY.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
        
    print(f"\n[STAGE 5 COMPLETE] All 10 CoTOP pilot runs completed. Summary: {summary_path}")


if __name__ == "__main__":
    main()
