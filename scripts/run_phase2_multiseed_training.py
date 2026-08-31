import os
import json
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import copy
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from envs.frozen_vec_env import FrozenVECEnv
from envs.entities import SimulationConfig
from models.a3c_agent import ActorCritic
from models.baselines.ddqn_agent import DDQNAgent
from utils.seed import set_seed
from utils.realization import get_git_sha

def hash_file(path):
    import hashlib
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def evaluate_agent(agent_name, agent, env, seed):
    """Deterministically evaluates an agent and returns disaggregated telemetry."""
    if agent_name == "CoTOP":
        agent.eval()
    
    obs, _ = env.reset(seed=seed)
    done = False
    
    eval_delays = []
    eval_energies = []
    
    tasks_generated = 0
    tasks_completed = 0
    tasks_failed = 0
    
    while not done:
        with torch.no_grad():
            if agent_name == "CoTOP":
                state = torch.FloatTensor(obs).unsqueeze(0)
                logits, _ = agent(state)
                mask = env.get_action_mask()
                mask_tensor = torch.BoolTensor(mask).unsqueeze(0)
                logits[~mask_tensor] = -1e9
                action = torch.argmax(logits, dim=-1).item()
            elif agent_name == "DDQN":
                mask = env.get_action_mask()
                action = agent.select_action(obs, action_mask=mask, deterministic=True)
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        tasks_generated += 1
        eval_delays.append(info.get('delay', 0.0))
        eval_energies.append(info.get('energy', 0.0))
            
        if info.get('completed', False):
            tasks_completed += 1
        else:
            tasks_failed += 1
                
    return {
        "mean_delay": float(np.mean(eval_delays)),
        "mean_energy": float(np.mean(eval_energies)),
        "tasks_generated": tasks_generated,
        "tasks_completed": tasks_completed,
        "tasks_failed": tasks_failed,
    }

def run_single_experiment(algo, geom, workload, seed, episodes=50):
    output_dir = f"results/phase2_multiseed/{algo}/{geom}_w{workload}_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)
    
    realization_path = f"data/evaluation_realizations/realization_{geom}_w{workload}_{seed}.json"
    real_hash = hash_file(realization_path)
    
    with open("configs/paper_parameters.yaml", 'r') as f:
        config_data = yaml.safe_load(f)
    config_data["num_tasks_per_vehicle_range"] = [workload, workload]
    config = SimulationConfig(**config_data)
    
    # Save config hash
    with open(f"{output_dir}/config.yaml", "w") as f:
        yaml.dump(config_data, f)
    config_hash = hash_file(f"{output_dir}/config.yaml")
    
    env = FrozenVECEnv(config=config, realization_path=realization_path)
    
    input_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    set_seed(seed)
    
    if algo == "CoTOP":
        agent = ActorCritic(input_dim, num_actions)
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.0002)
    elif algo == "DDQN":
        agent = DDQNAgent(input_dim=input_dim, num_actions=num_actions, gamma=0.99, learning_rate=0.0002, epsilon_decay_episodes=40)
    else:
        raise ValueError("Unknown algo")
        
    training_metrics = []
    gamma = 0.99
    
    try:
        # ----------------------------------------
        # Training Loop
        # ----------------------------------------
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed)
            done = False
            rewards = []
            
            if algo == "CoTOP":
                values, log_probs = [], []
                while not done:
                    state = torch.FloatTensor(obs).unsqueeze(0)
                    logits, value = agent(state)
                    mask = env.get_action_mask()
                    mask_tensor = torch.BoolTensor(mask).unsqueeze(0)
                    logits[~mask_tensor] = -1e9
                    probs = F.softmax(logits, dim=-1)
                    m = torch.distributions.Categorical(probs)
                    action = m.sample()
                    
                    obs, reward, term, trunc, info = env.step(action.item())
                    done = term or trunc
                    values.append(value)
                    log_probs.append(m.log_prob(action))
                    rewards.append(reward)
                    
                R = 0
                returns = []
                for r in rewards[::-1]:
                    R = r + gamma * R
                    returns.insert(0, R)
                returns = torch.FloatTensor(returns).unsqueeze(1)
                values = torch.cat(values)
                log_probs = torch.cat(log_probs)
                advantages = returns - values.detach()
                actor_loss = -(log_probs * advantages.squeeze()).mean()
                critic_loss = F.mse_loss(values, returns)
                optimizer.zero_grad()
                (actor_loss + 0.5 * critic_loss).backward()
                optimizer.step()
                
                training_metrics.append({
                    "episode": episode,
                    "reward": sum(rewards),
                    "actor_loss": float(actor_loss.item()),
                    "critic_loss": float(critic_loss.item())
                })
                
            elif algo == "DDQN":
                agent.set_episode(episode)
                while not done:
                    mask = env.get_action_mask()
                    action = agent.select_action(obs, action_mask=mask, deterministic=False)
                    next_obs, reward, term, trunc, info = env.step(action)
                    done = term or trunc
                    agent.store_transition(obs, action, reward, next_obs, done, env.get_action_mask())
                    agent.update()
                    obs = next_obs
                    rewards.append(reward)
                    
                training_metrics.append({
                    "episode": episode,
                    "reward": sum(rewards)
                })

        # Save Checkpoint
        ckpt_path = f"{output_dir}/checkpoint.pt"
        if algo == "CoTOP":
            torch.save(agent.state_dict(), ckpt_path)
        else:
            torch.save(agent.online_net.state_dict(), ckpt_path)
        model_hash = hash_file(ckpt_path)
        
        # Save training metrics
        with open(f"{output_dir}/training_metrics.json", "w") as f:
            json.dump(training_metrics, f)
            
        # ----------------------------------------
        # Evaluation
        # ----------------------------------------
        eval_metrics = evaluate_agent(algo, agent, env, seed)
        with open(f"{output_dir}/evaluation_metrics.json", "w") as f:
            json.dump(eval_metrics, f)
            
        # Manifests
        realization_manifest = {
            "realization_hash": real_hash,
            "path": realization_path
        }
        with open(f"{output_dir}/realization_manifest.json", "w") as f:
            json.dump(realization_manifest, f)
            
        run_manifest = {
            "git_sha": get_git_sha(),
            "config_hash": config_hash,
            "realization_hash": real_hash,
            "model_hash": model_hash,
            "seed": seed,
            "algorithm": algo,
            "workload": f"w{workload}",
            "geometry": geom,
            "status": "SUCCESS"
        }
        with open(f"{output_dir}/run_manifest.json", "w") as f:
            json.dump(run_manifest, f)
            
        return run_manifest, eval_metrics, None
        
    except Exception as e:
        run_manifest = {
            "git_sha": get_git_sha(),
            "seed": seed,
            "algorithm": algo,
            "workload": f"w{workload}",
            "geometry": geom,
            "status": "FAILED",
            "error": str(e)
        }
        with open(f"{output_dir}/run_manifest.json", "w") as f:
            json.dump(run_manifest, f)
        return run_manifest, None, str(e)


def main():
    geometries = ["corridor_2400m", "grid_200m"]
    workloads = [20, 30, 40]
    seeds = [42, 43, 44, 45, 46]
    algorithms = ["CoTOP", "DDQN"]
    
    tasks = []
    for geom in geometries:
        for workload in workloads:
            for seed in seeds:
                for algo in algorithms:
                    tasks.append((algo, geom, workload, seed))
                    
    results_list = []
    
    print(f"Starting execution of {len(tasks)} runs...")
    
    # Run sequentially due to environment limits or in parallel if safe
    # Given FrozenVECEnv is CPU-bound purely via numpy/torch, we can use ProcessPoolExecutor
    # BUT we will just run it sequentially for ultimate stability on Windows
    
    for i, (algo, geom, workload, seed) in enumerate(tasks):
        print(f"Run {i+1}/{len(tasks)}: {algo} | {geom} | w{workload} | seed {seed}")
        manifest, evals, err = run_single_experiment(algo, geom, workload, seed)
        if err:
            print(f"  -> FAILED: {err}")
            results_list.append({
                "Algorithm": algo,
                "Geometry": geom,
                "Workload": f"w{workload}",
                "Seed": seed,
                "Status": "FAILED",
                "MeanDelay": None,
                "MeanEnergy": None,
                "Completed": None,
                "Error": err
            })
        else:
            print(f"  -> SUCCESS | Delay: {evals['mean_delay']:.4f} | Energy: {evals['mean_energy']:.4f}")
            results_list.append({
                "Algorithm": algo,
                "Geometry": geom,
                "Workload": f"w{workload}",
                "Seed": seed,
                "Status": "SUCCESS",
                "MeanDelay": evals["mean_delay"],
                "MeanEnergy": evals["mean_energy"],
                "Completed": evals["tasks_completed"],
                "Error": None
            })
            
    df = pd.DataFrame(results_list)
    df.to_csv("results/phase2_multiseed/seed_results.csv", index=False)
    
    # Generate MD
    success_count = len(df[df['Status'] == 'SUCCESS'])
    md_content = f"""# PHASE 2: MULTISEED TRAINING EXPERIMENT

## 1. Experiment Design
- **Algorithms**: CoTOP, DDQN
- **Geometries**: corridor_2400m, grid_200m
- **Workloads**: w20, w30, w40
- **Seeds**: 42, 43, 44, 45, 46
- **Total Planned Runs**: 60
- **Total Successful Runs**: {success_count}

## 2. Invariants Assured
- Strict use of FrozenVECEnv across all runs
- Absolute pairing of exogenous traces per seed/geometry/workload configuration
- Automated hash locking of configuration, model weights, and realization files

## 3. Results Summary
Please refer to `results/phase2_multiseed/seed_results.csv` for complete tabular data.
"""
    with open("docs/PHASE2_MULTISEED_TRAINING.md", "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"Done. Successfully processed {success_count}/{len(tasks)} runs.")

if __name__ == "__main__":
    main()
