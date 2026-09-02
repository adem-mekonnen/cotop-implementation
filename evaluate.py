import argparse
import os
import sys
import json
import hashlib
import torch
import numpy as np
import yaml
import pandas as pd

from envs.vec_env import VECEnv
from envs.frozen_vec_env import FrozenVECEnv
from envs.entities import SimulationConfig
from models.a3c_agent import ActorCritic
from models.baselines.ddqn_agent import QNetwork
from models.baselines.local import LocalPolicy
from models.baselines.greedy import GreedyPolicy
from utils.seed import set_seed
from utils.checkpoint_io import load_checkpoint_strict, compute_file_sha256, compute_model_param_hash

def evaluate():
    parser = argparse.ArgumentParser(description="Official CoTOP and Baseline Policy Evaluator")
    parser.add_argument('--mode', type=str, 
                        choices=['cotop', 'ddqn', 'local', 'greedy', 'wo_md', 'wo_tp', 'wo_co'], 
                        default='cotop')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default='configs/paper_parameters.yaml')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--scenario', type=str, default='corridor_2400m', choices=['corridor_2400m', 'grid_200m', 'linear_corridor', 'urban_manhattan'])
    parser.add_argument('--scenario_geometry', type=str, default=None)
    parser.add_argument('--priority_mode', type=str, default='paper_literal', choices=['paper_literal', 'normalized_candidate'])
    parser.add_argument('--coverage_mode', type=str, default='completion_position', choices=['completion_position', 'continuous_required_rsus'])
    parser.add_argument('--spatial_graph_radius', type=float, default=200.0)
    parser.add_argument('--max_vehicles', type=int, default=10)
    parser.add_argument('--workload', type=int, default=None, help="Tasks per vehicle override")
    parser.add_argument('--realization_path', type=str, default=None, help="Path to frozen realization JSON for deterministic evaluation")
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--manifest_path', type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)
        
    if args.workload is not None:
        yaml_config["num_tasks_per_vehicle_range"] = [args.workload, args.workload]
        
    config = SimulationConfig(**yaml_config)
    
    # Configure scenario & ablation flags
    scenario_name = args.scenario_geometry if args.scenario_geometry is not None else args.scenario
    use_mobility = (args.mode != 'wo_md')
    use_priority = (args.mode != 'wo_tp')
    sim_geom = "grid_200m" if scenario_name in ["grid_200m", "urban_manhattan"] else "corridor_2400m"
    
    # Initialize Environment
    is_frozen = bool(args.realization_path and os.path.exists(args.realization_path))
    if is_frozen:
        env = FrozenVECEnv(config=config, realization_path=args.realization_path)
    else:
        env = VECEnv(
            config=config, 
            port=9999, 
            scenario_geometry=sim_geom,
            use_mobility_model=use_mobility, 
            use_priority=use_priority,
            priority_mode=args.priority_mode,
            coverage_mode=args.coverage_mode,
            spatial_graph_radius=args.spatial_graph_radius,
            max_vehicles=args.max_vehicles,
            seed=args.seed
        )
    
    input_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    ckpt_metadata = {}
    model = None
    policy = None

    # Initialize Policies & Strict Checkpoint Ingestion
    if args.mode in ['cotop', 'wo_md', 'wo_tp']:
        model = ActorCritic(input_dim, num_actions)
        ckpt_path = args.checkpoint_path if args.checkpoint_path else "results/checkpoints/a3c_agent.pth"
        ckpt_metadata = load_checkpoint_strict(
            checkpoint_path=ckpt_path,
            model=model,
            expected_algorithm="CoTOP" if args.mode == "cotop" else None,
            device="cpu"
        )
        print(f"[INFO] Loaded verified CoTOP checkpoint: {ckpt_path} (SHA: {ckpt_metadata['checkpoint_sha256'][:12]}...)")
        
    elif args.mode == 'ddqn':
        model = QNetwork(input_dim, num_actions)
        ckpt_path = args.checkpoint_path if args.checkpoint_path else "results/checkpoints/ddqn_agent.pth"
        ckpt_metadata = load_checkpoint_strict(
            checkpoint_path=ckpt_path,
            model=model,
            expected_algorithm="DDQN",
            device="cpu"
        )
        print(f"[INFO] Loaded verified DDQN checkpoint: {ckpt_path} (SHA: {ckpt_metadata['checkpoint_sha256'][:12]}...)")
        
    elif args.mode in ['local', 'wo_co']:
        policy = LocalPolicy(config=config)
    elif args.mode == 'greedy':
        policy = GreedyPolicy(config=config)

    total_rewards = []
    total_delays = []
    total_energies = []
    total_comm_delays = []
    total_comp_delays = []
    total_wait_delays = []
    action_sequence = []
    all_task_records = []
    
    total_completed_tasks = 0
    total_evaluated_tasks = 0

    print(f"=== Starting Evaluation ===")
    print(f"Mode: {args.mode} | Geometry: {sim_geom} | Frozen Replay: {is_frozen} | Episodes: {args.episodes}")

    num_episodes = 1 if is_frozen else args.episodes

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        
        ep_reward = 0
        ep_delay = 0
        ep_energy = 0
        ep_completed = 0
        ep_tasks = 0
        
        while not done:
            mask = env.get_action_mask()
            
            if policy is not None:
                action = policy.select_action(obs)
            else:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    if args.mode == 'ddqn':
                        logits = model(obs_tensor)
                    else:
                        logits, _ = model(obs_tensor)
                mask_tensor = torch.BoolTensor(mask).unsqueeze(0)
                logits[~mask_tensor] = -1e9
                action = torch.argmax(logits, dim=-1).item()
                
            action_sequence.append(int(action))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            d = info.get('delay', 0.0)
            e = info.get('energy', 0.0)
            c_d = info.get('comm_delay', 0.0)
            cp_d = info.get('comp_delay', 0.0)
            w_d = info.get('wait_delay', 0.0)
            comp = info.get('completed', False)
            f_reason = info.get('failure_reason', 'NONE')
            
            ep_reward += reward
            ep_tasks += 1
            ep_delay += d
            ep_energy += e
            
            total_comm_delays.append(c_d)
            total_comp_delays.append(cp_d)
            total_wait_delays.append(w_d)
            
            if comp:
                ep_completed += 1
                
            all_task_records.append({
                "task_index": len(all_task_records),
                "action": action,
                "case": info.get("case", 1),
                "delay_s": d,
                "energy_j": e,
                "comm_delay_s": c_d,
                "comp_delay_s": cp_d,
                "wait_delay_s": w_d,
                "completed": comp,
                "failure_reason": f_reason
            })
                
        total_rewards.append(ep_reward)
        total_delays.append(ep_delay / max(ep_tasks, 1))
        total_energies.append(ep_energy / max(ep_tasks, 1))
        total_completed_tasks += ep_completed
        total_evaluated_tasks += ep_tasks
        
        comp_ratio_ep = (ep_completed / ep_tasks) if ep_tasks > 0 else 0.0
        print(f"Episode {episode+1:02d} | Reward: {ep_reward:6.2f} | Avg Delay: {total_delays[-1]:5.4f}s | Avg Energy: {total_energies[-1]:5.4f}J | Comp Ratio: {comp_ratio_ep * 100:.1f}%")

    overall_completion_ratio = (total_completed_tasks / total_evaluated_tasks) if total_evaluated_tasks > 0 else 0.0
    action_seq_hash = hashlib.sha256(json.dumps(action_sequence).encode()).hexdigest()
    
    print("=" * 50)
    print(f"       FINAL EVALUATION REPORT ({args.mode.upper()})")
    print("=" * 50)
    print(f"Average Reward:           {np.mean(total_rewards):.4f} ± {np.std(total_rewards):.4f}")
    print(f"Average Delay (s):        {np.mean(total_delays):.4f} ± {np.std(total_delays):.4f}")
    print(f"Average Energy (J):       {np.mean(total_energies):.4f} ± {np.std(total_energies):.4f}")
    print(f"Task Completion Ratio:    {overall_completion_ratio * 100:.2f}% ({total_completed_tasks}/{total_evaluated_tasks})")
    print(f"Action Sequence SHA-256:  {action_seq_hash}")
    print("=" * 50)
    
    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        pd.DataFrame(all_task_records).to_csv(args.output_csv, index=False)
        print(f"[INFO] Saved task-level evaluation trace to {args.output_csv}")

    if args.manifest_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.manifest_path)), exist_ok=True)
        manifest_data = {
            "algorithm": args.mode.upper(),
            "mode": args.mode,
            "scenario": sim_geom,
            "workload": args.workload if args.workload else config.num_tasks_per_vehicle_range[0],
            "seed": args.seed,
            "episodes": num_episodes,
            "tasks_generated": total_evaluated_tasks,
            "tasks_completed": total_completed_tasks,
            "tasks_failed": total_evaluated_tasks - total_completed_tasks,
            "completion_ratio": overall_completion_ratio,
            "mean_delay_s": float(np.mean(total_delays)),
            "mean_energy_j": float(np.mean(total_energies)),
            "comm_delay_s": float(np.mean(total_comm_delays)) if total_comm_delays else 0.0,
            "comp_delay_s": float(np.mean(total_comp_delays)) if total_comp_delays else 0.0,
            "wait_delay_s": float(np.mean(total_wait_delays)) if total_wait_delays else 0.0,
            "action_sequence_sha256": action_seq_hash,
            "checkpoint_metadata": ckpt_metadata,
            "realization_path": args.realization_path,
            "realization_sha256": compute_file_sha256(args.realization_path) if is_frozen else "N/A"
        }
        with open(args.manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2, default=str)
        print(f"[INFO] Saved evaluation manifest to {args.manifest_path}")

    env.close()

if __name__ == '__main__':
    evaluate()