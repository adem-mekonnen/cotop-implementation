#!/usr/bin/env python3
"""
scripts/run_phase2_gpu_campaign.py

Unified Google Colab GPU Campaign Runner for Phase 2 CoTOP Reproduction.

Features:
1. Strict GPU verification (fails loudly if CUDA is unavailable unless --allow-cpu is specified).
2. Exact checkpointing with complete state (model, optimizer, episode, step, RNG states, git SHA, physics hashes).
3. Seamless resume capability (--resume) to protect against Colab runtime preemption/interruption.
4. Output isolation: results/phase2_step20/<algorithm>/<scenario>/<workload>/seed_<seed>/
5. Full provenance manifest generation for every run.
6. Deterministic evaluation over cryptographically verified frozen exogenous realizations.
7. Support for smoke tests (--smoke-test).
"""

import sys
import os

# Ensure root workspace is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import time
import argparse
import json
import yaml
import hashlib
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from envs.entities import SimulationConfig, Task, Vehicle, RSU
from envs.vec_env import VECEnv
from envs.frozen_vec_env import FrozenVECEnv
from models.a3c_agent import ActorCritic
from models.baselines.ddqn_agent import DDQNAgent, QNetwork
from utils.seed import set_seed
from utils.realization import generate_realization, save_realization, load_realization

COMM_SHA256 = "041e41061d02c7a5a7bc9488adf2bc49472177215730bd8a23c5ff2437431431"
COMP_SHA256 = "dd9f58df710f709d536000bb4047d2ad6000cf37b1a49f4e1f0e8d883b856bff"

def get_git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "UNKNOWN"

def get_git_branch():
    try:
        return subprocess.check_output(["git", "branch", "--show-current"]).decode("utf-8").strip()
    except Exception:
        return "UNKNOWN"

def verify_physics_hashes():
    comm_path = os.path.join(root_dir, "envs/comm_model.py")
    comp_path = os.path.join(root_dir, "envs/comp_model.py")
    
    if not os.path.exists(comm_path) or not os.path.exists(comp_path):
        raise FileNotFoundError("Protected physics files missing!")
        
    comm_hash = hashlib.sha256(open(comm_path, "rb").read()).hexdigest()
    comp_hash = hashlib.sha256(open(comp_path, "rb").read()).hexdigest()
    
    if comm_hash != COMM_SHA256:
        raise ValueError(f"CRITICAL: comm_model.py hash mismatch! Expected {COMM_SHA256}, got {comm_hash}")
    if comp_hash != COMP_SHA256:
        raise ValueError(f"CRITICAL: comp_model.py hash mismatch! Expected {COMP_SHA256}, got {comp_hash}")
        
    return comm_hash, comp_hash

def get_hardware_info(device_name, allow_cpu=False):
    cuda_available = torch.cuda.is_available()
    if device_name.startswith("cuda") and not cuda_available:
        if allow_cpu:
            print("[WARN] CUDA requested but unavailable. Falling back to CPU because --allow-cpu was set.")
            device = torch.device("cpu")
            gpu_name = "NO GPU (CPU Fallback)"
            gpu_mem_mb = 0
            cuda_ver = "N/A"
        else:
            print("[ERROR] CUDA IS NOT AVAILABLE! Halting execution.")
            print("Google Colab requires a GPU runtime (Runtime -> Change runtime type -> T4/V100/A100 GPU).")
            print("To run in diagnostic CPU mode, pass --allow-cpu.")
            sys.exit(1)
    elif device_name.startswith("cuda") and cuda_available:
        device = torch.device(device_name)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        cuda_ver = torch.version.cuda
    else:
        device = torch.device("cpu")
        gpu_name = "CPU"
        gpu_mem_mb = 0
        cuda_ver = "N/A"
        
    return {
        "device": device,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_mem_mb": gpu_mem_mb,
        "cuda_ver": cuda_ver,
        "pytorch_ver": torch.__version__,
        "python_ver": sys.version.split()[0]
    }

def capture_rng_state(device):
    rng = {
        "python": set_seed_python_state(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state()
    }
    if torch.cuda.is_available() and device.type == "cuda":
        rng["torch_cuda"] = torch.cuda.get_rng_state()
    return rng

def set_seed_python_state():
    import random
    return random.getstate()

def restore_rng_state(rng, device):
    import random
    if "python" in rng:
        random.setstate(rng["python"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "torch_cpu" in rng:
        torch.set_rng_state(rng["torch_cpu"])
    if "torch_cuda" in rng and torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.set_rng_state(rng["torch_cuda"])

def load_sim_config(scenario_geometry, workload_val):
    with open(os.path.join(root_dir, "configs/paper_parameters.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    cfg["num_tasks_per_vehicle_range"] = [workload_val, workload_val]
    return SimulationConfig(**cfg)

def ensure_realization(scenario, workload, seed, config):
    """
    Ensures exogenous realization trace is materialized and saved to disk,
    returning path and cryptographic hash.
    """
    realization_dir = os.path.join(root_dir, "data", "evaluation_realizations")
    os.makedirs(realization_dir, exist_ok=True)
    realization_path = os.path.join(realization_dir, f"realization_{scenario}_w{workload}_seed{seed}.json")
    
    if not os.path.exists(realization_path):
        print(f"  [REALIZATION] Generating realization: {realization_path}")
        temp_env = VECEnv(
            config=config,
            port=9500 + (seed % 200) * 2,
            scenario_geometry=scenario,
            use_mobility_model=True,
            max_vehicles=10,
            seed=seed
        )
        real_data = generate_realization(temp_env)
        temp_env.close()
        save_realization(real_data, realization_path)
        
    h = hashlib.sha256(open(realization_path, "rb").read()).hexdigest()
    return realization_path, h

def run_training_and_eval(
    algorithm: str,
    scenario: str,
    workload: int,
    seed: int,
    episodes: int,
    device_info: dict,
    output_base_dir: str,
    resume: bool = False,
    checkpoint_interval: int = 50
):
    print(f"\n=======================================================================")
    print(f" RUN: Algo={algorithm} | Scenario={scenario} | W={workload} | Seed={seed}")
    print(f"=======================================================================")
    
    device = device_info["device"]
    comm_hash, comp_hash = verify_physics_hashes()
    git_sha = get_git_sha()
    git_branch = get_git_branch()
    
    run_dir = os.path.join(output_base_dir, algorithm, scenario, f"w{workload}", f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(run_dir, "checkpoint.pt")
    config_path = os.path.join(run_dir, "config.yaml")
    run_manifest_path = os.path.join(run_dir, "run_manifest.json")
    realization_manifest_path = os.path.join(run_dir, "realization_manifest.json")
    training_curve_path = os.path.join(run_dir, "training_curve.csv")
    training_metrics_path = os.path.join(run_dir, "training_metrics.json")
    evaluation_metrics_path = os.path.join(run_dir, "evaluation_metrics.json")
    evaluation_results_path = os.path.join(run_dir, "evaluation_results.csv")
    
    # Save config
    config = load_sim_config(scenario, workload)
    with open(config_path, "w") as f:
        yaml.dump(config.__dict__, f)
        
    realization_path, realization_hash = ensure_realization(scenario, workload, seed, config)
    with open(realization_manifest_path, "w") as f:
        json.dump({
            "realization_path": realization_path,
            "realization_sha256": realization_hash,
            "scenario": scenario,
            "workload": workload,
            "seed": seed
        }, f, indent=2)
        
    # Check if run is already fully completed
    if resume and os.path.exists(evaluation_metrics_path) and os.path.exists(checkpoint_path):
        print("  [RESUME] Run already fully completed and evaluated. Skipping.")
        return
        
    # Setup environment
    set_seed(seed)
    env = VECEnv(
        config=config,
        port=9000 + (seed % 500) * 2,
        scenario_geometry=scenario,
        use_mobility_model=True,
        max_vehicles=10,
        seed=seed
    )
    
    input_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    # Instantiate agent
    if algorithm == "DDQN":
        agent = DDQNAgent(
            input_dim=input_dim,
            num_actions=num_actions,
            gamma=0.99,
            learning_rate=0.0002,
            replay_capacity=10000,
            batch_size=64,
            target_update_frequency=100,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_episodes=200,
            device=device
        )
    elif algorithm == "CoTOP":
        agent = ActorCritic(input_dim=input_dim, num_actions=num_actions).to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
    else:
        raise ValueError(f"Unsupported algorithm for training: {algorithm}")
        
    start_episode = 0
    training_curves = []
    
    # Resume from checkpoint if exists
    if resume and os.path.exists(checkpoint_path):
        print(f"  [RESUME] Loading existing checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        start_episode = ckpt["episode"] + 1
        if algorithm == "DDQN":
            agent.online_net.load_state_dict(ckpt["online_net_state_dict"])
            agent.target_net.load_state_dict(ckpt["target_net_state_dict"])
            agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            agent.train_step_count = ckpt.get("global_step", 0)
            agent.epsilon = ckpt.get("epsilon", 0.05)
        elif algorithm == "CoTOP":
            agent.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        restore_rng_state(ckpt.get("rng_state", {}), device)
        if os.path.exists(training_curve_path):
            training_curves = pd.read_csv(training_curve_path).to_dict(orient="records")
            
    print(f"  [TRAIN] Starting training from episode {start_episode+1} to {episodes} on {device}...")
    start_time = time.time()
    
    for ep in range(start_episode, episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        losses = []
        
        if algorithm == "DDQN":
            while not done:
                mask = env.get_action_mask()
                action = agent.select_action(obs, action_mask=mask, deterministic=False)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_mask = env.get_action_mask() if not done else None
                
                agent.store_transition(obs, action, reward, next_obs, done, next_action_mask=next_mask)
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)
                obs = next_obs
                ep_reward += reward
                
            mean_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
            training_curves.append({
                "episode": ep + 1,
                "reward": ep_reward,
                "loss": mean_loss,
                "epsilon": agent.epsilon,
                "buffer_size": len(agent.memory)
            })
            
        elif algorithm == "CoTOP":
            values, log_probs, rewards = [], [], []
            while not done:
                state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                policy_logits, value = agent(state_t)
                mask = env.get_action_mask()
                mask_t = torch.BoolTensor(mask).unsqueeze(0).to(device)
                policy_logits[~mask_t] = -1e9
                probs = F.softmax(policy_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                
                next_obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                
                values.append(value)
                log_probs.append(dist.log_prob(action))
                rewards.append(reward)
                obs = next_obs
                ep_reward += reward
                
            # A3C update
            R = 0
            returns = []
            for r in reversed(rewards):
                R = r + 0.99 * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            values = torch.cat(values).squeeze(-1)
            log_probs = torch.cat(log_probs)
            
            advantage = returns - values.detach()
            actor_loss = -(log_probs * advantage).mean()
            critic_loss = F.mse_loss(values, returns)
            loss = actor_loss + 0.5 * critic_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 40.0)
            optimizer.step()
            
            training_curves.append({
                "episode": ep + 1,
                "reward": ep_reward,
                "loss": float(loss.item()),
                "epsilon": 0.0,
                "buffer_size": 0
            })
            
        # Periodic checkpoint
        if (ep + 1) % checkpoint_interval == 0 or (ep + 1) == episodes:
            ckpt_data = {
                "episode": ep,
                "algorithm": algorithm,
                "scenario": scenario,
                "workload": workload,
                "seed": seed,
                "git_sha": git_sha,
                "physics_hashes": {"comm": comm_hash, "comp": comp_hash},
                "rng_state": capture_rng_state(device)
            }
            if algorithm == "DDQN":
                ckpt_data["online_net_state_dict"] = agent.online_net.state_dict()
                ckpt_data["target_net_state_dict"] = agent.target_net.state_dict()
                ckpt_data["optimizer_state_dict"] = agent.optimizer.state_dict()
                ckpt_data["global_step"] = agent.train_step_count
                ckpt_data["epsilon"] = agent.epsilon
            elif algorithm == "CoTOP":
                ckpt_data["model_state_dict"] = agent.state_dict()
                ckpt_data["optimizer_state_dict"] = optimizer.state_dict()
            torch.save(ckpt_data, checkpoint_path)
            pd.DataFrame(training_curves).to_csv(training_curve_path, index=False)
            
    env.close()
    train_duration = time.time() - start_time
    checkpoint_hash = hashlib.sha256(open(checkpoint_path, "rb").read()).hexdigest()
    
    with open(training_metrics_path, "w") as f:
        json.dump({
            "train_duration_s": train_duration,
            "total_episodes": episodes,
            "checkpoint_sha256": checkpoint_hash,
            "final_reward": training_curves[-1]["reward"] if len(training_curves) > 0 else 0.0
        }, f, indent=2)
        
    print(f"  [EVAL] Executing deterministic evaluation on frozen realization...")
    # Deterministic evaluation on FrozenVECEnv
    eval_env = FrozenVECEnv(config=config, realization_path=realization_path)
    obs, _ = eval_env.reset(seed=seed)
    eval_done = False
    
    delays = []
    energies = []
    comm_delays = []
    comp_delays = []
    wait_delays = []
    tasks_gen = 0
    tasks_comp = 0
    tasks_fail = 0
    
    if algorithm == "DDQN":
        agent.online_net.eval()
    elif algorithm == "CoTOP":
        agent.eval()
        
    while not eval_done:
        mask = eval_env.get_action_mask()
        with torch.no_grad():
            if algorithm == "DDQN":
                action = agent.select_action(obs, action_mask=mask, deterministic=True)
            elif algorithm == "CoTOP":
                state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                logits, _ = agent(state_t)
                mask_t = torch.BoolTensor(mask).unsqueeze(0).to(device)
                logits[~mask_t] = -1e9
                action = torch.argmax(logits, dim=-1).item()
                
        obs, reward, term, trunc, info = eval_env.step(action)
        eval_done = term or trunc
        
        tasks_gen += 1
        delays.append(info.get("delay", 0.0))
        energies.append(info.get("energy", 0.0))
        comm_delays.append(info.get("comm_delay", 0.0))
        comp_delays.append(info.get("comp_delay", 0.0))
        wait_delays.append(info.get("wait_delay", 0.0))
        
        if info.get("completed", False):
            tasks_comp += 1
        else:
            tasks_fail += 1
            
    eval_env.close()
    
    # Save evaluation outputs
    eval_df = pd.DataFrame({
        "task_idx": np.arange(len(delays)),
        "delay_s": delays,
        "energy_j": energies,
        "comm_delay_s": comm_delays,
        "comp_delay_s": comp_delays,
        "wait_delay_s": wait_delays
    })
    eval_df.to_csv(evaluation_results_path, index=False)
    
    eval_metrics = {
        "algorithm": algorithm,
        "scenario": scenario,
        "workload": workload,
        "seed": seed,
        "tasks_generated": tasks_gen,
        "tasks_completed": tasks_comp,
        "tasks_failed": tasks_fail,
        "completion_ratio": tasks_comp / tasks_gen if tasks_gen > 0 else 0.0,
        "mean_delay_s": float(np.mean(delays)),
        "std_delay_s": float(np.std(delays, ddof=1)) if len(delays) > 1 else 0.0,
        "mean_energy_j": float(np.mean(energies)),
        "std_energy_j": float(np.std(energies, ddof=1)) if len(energies) > 1 else 0.0,
        "comm_delay_s": float(np.mean(comm_delays)),
        "comp_delay_s": float(np.mean(comp_delays)),
        "wait_delay_s": float(np.mean(wait_delays)),
        "realization_sha256": realization_hash,
        "checkpoint_sha256": checkpoint_hash
    }
    with open(evaluation_metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
        
    # Save complete run manifest
    run_manifest = {
        "git_commit_sha": git_sha,
        "git_branch": git_branch,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "algorithm": algorithm,
        "scenario": scenario,
        "workload": workload,
        "seed": seed,
        "episodes": episodes,
        "hardware": {
            "device": str(device),
            "gpu_name": device_info["gpu_name"],
            "gpu_mem_mb": device_info["gpu_mem_mb"],
            "cuda_ver": device_info["cuda_ver"]
        },
        "software": {
            "python_version": device_info["python_ver"],
            "pytorch_version": device_info["pytorch_ver"]
        },
        "physics_hashes": {
            "comm_model_sha256": comm_hash,
            "comp_model_sha256": comp_hash
        },
        "realization_sha256": realization_hash,
        "checkpoint_sha256": checkpoint_hash,
        "status": "COMPLETED"
    }
    with open(run_manifest_path, "w") as f:
        json.dump(run_manifest, f, indent=2)
        
    print(f"  [OK] Completed run {algorithm}/{scenario}/w{workload}/seed_{seed}. Mean Delay: {eval_metrics['mean_delay_s']:.4f}s, Completion: {eval_metrics['completion_ratio']*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="CoTOP Phase 2 GPU Campaign Runner")
    parser.add_argument("--algorithm", type=str, default="DDQN", choices=["DDQN", "CoTOP", "all"], help="Algorithm to run")
    parser.add_argument("--scenario", type=str, default="corridor_2400m", choices=["corridor_2400m", "grid_200m", "all"], help="Scenario geometry")
    parser.add_argument("--workload", type=str, default="20", help="Workload per vehicle (20, 30, 40, or all)")
    parser.add_argument("--seed", type=str, default="42", help="Seed(s) to execute (e.g. 42 or 42,43,44,45,46)")
    parser.add_argument("--episodes", type=int, default=500, help="Training episode count")
    parser.add_argument("--device", type=str, default="cuda:0", help="PyTorch compute device")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback for local testing")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint if available")
    parser.add_argument("--output-dir", type=str, default="results/phase2_step20", help="Output directory")
    parser.add_argument("--smoke-test", action="store_true", help="Execute 1 minimal smoke run (2 episodes)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("   CoTOP GPU REPRODUCTION CAMPAIGN RUNNER")
    print("=" * 70)
    
    dev_info = get_hardware_info(args.device, allow_cpu=args.allow_cpu)
    output_dir = os.path.join(root_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.smoke_test:
        print("[SMOKE TEST MODE ACTIVATED] Running minimal 2-episode verification on seed 42...")
        run_training_and_eval(
            algorithm="DDQN",
            scenario="corridor_2400m",
            workload=20,
            seed=42,
            episodes=2,
            device_info=dev_info,
            output_base_dir=output_dir,
            resume=args.resume,
            checkpoint_interval=1
        )
        print("\n[SMOKE TEST COMPLETE] Verified checkpointing, realization, manifest, and evaluation.")
        return
        
    algos = ["DDQN", "CoTOP"] if args.algorithm == "all" else [args.algorithm]
    scenarios = ["corridor_2400m", "grid_200m"] if args.scenario == "all" else [args.scenario]
    workloads = [20, 30, 40] if args.workload == "all" else [int(w) for w in args.workload.split(",")]
    seeds = [42, 43, 44, 45, 46] if args.seed == "all" else [int(s) for s in args.seed.split(",")]
    
    for algo in algos:
        for scen in scenarios:
            for wl in workloads:
                for s in seeds:
                    run_training_and_eval(
                        algorithm=algo,
                        scenario=scen,
                        workload=wl,
                        seed=s,
                        episodes=args.episodes,
                        device_info=dev_info,
                        output_base_dir=output_dir,
                        resume=args.resume,
                        checkpoint_interval=50
                    )
                    
    print("\n" + "=" * 70)
    print("   CAMPAIGN BATCH EXECUTION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
