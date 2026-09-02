import argparse
import os
import torch
import numpy as np
import yaml

from envs.vec_env import VECEnv
from envs.entities import SimulationConfig
from models.a3c_agent import ActorCritic
from models.baselines.local import LocalPolicy
from models.baselines.greedy import GreedyPolicy
from utils.seed import set_seed

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, 
                        choices=['cotop', 'ddqn', 'local', 'greedy', 'wo_md', 'wo_tp', 'wo_co'], 
                        default='cotop')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default='configs/paper_parameters.yaml')
    parser.add_argument('--checkpoint_path', type=str, default='results/checkpoints/a3c_agent.pth')
    parser.add_argument('--scenario_geometry', type=str, default='corridor_2400m', choices=['corridor_2400m', 'grid_200m', 'linear_corridor', 'urban_manhattan'])
    parser.add_argument('--priority_mode', type=str, default='paper_literal', choices=['paper_literal', 'normalized_candidate'])
    parser.add_argument('--coverage_mode', type=str, default='completion_position', choices=['completion_position', 'continuous_required_rsus'])
    parser.add_argument('--spatial_graph_radius', type=float, default=200.0)
    parser.add_argument('--max_vehicles', type=int, default=10)
    parser.add_argument('--workload', type=int, default=None, help="Tasks per vehicle override")
    parser.add_argument('--output_csv', type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)
        
    if args.workload is not None:
        yaml_config["num_tasks_per_vehicle_range"] = [args.workload, args.workload]
        
    config = SimulationConfig(**yaml_config)
    
    # Configure ablation flags cleanly
    use_mobility = (args.mode != 'wo_md')
    use_priority = (args.mode != 'wo_tp')
    sim_geom = "grid_200m" if args.scenario_geometry in ["grid_200m", "urban_manhattan"] else "corridor_2400m"
    
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
    
    # Initialize Policies
    if args.mode in ['cotop', 'wo_md', 'wo_tp']:
        model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
        ckpt_path = args.checkpoint_path
        if os.path.exists(ckpt_path):
            try:
                ckpt_data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
                    model.load_state_dict(ckpt_data["model_state_dict"])
                else:
                    model.load_state_dict(ckpt_data)
                print(f"[INFO] Loaded trained CoTOP model from {ckpt_path}.")
            except Exception as e:
                print(f"[WARN] Error loading checkpoint ({e}). Using initialized weights.")
        else:
            print(f"[WARN] Checkpoint not found at {ckpt_path}. Evaluating with untrained agent.")
        model.eval()
        policy = None
    elif args.mode == 'ddqn':
        from models.baselines.ddqn_agent import QNetwork
        model = QNetwork(env.observation_space.shape[0], env.action_space.n)
        ckpt_path = args.checkpoint_path
        if os.path.exists(ckpt_path):
            try:
                ckpt_data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                if isinstance(ckpt_data, dict) and "online_net_state_dict" in ckpt_data:
                    model.load_state_dict(ckpt_data["online_net_state_dict"])
                elif isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
                    model.load_state_dict(ckpt_data["model_state_dict"])
                else:
                    model.load_state_dict(ckpt_data)
                print(f"[INFO] Loaded trained DDQN model from {ckpt_path}.")
            except Exception as e:
                print(f"[WARN] Error loading checkpoint ({e}). Using initialized weights.")
        else:
            print(f"[WARN] Checkpoint not found at {ckpt_path}. Evaluating with untrained DDQN agent.")
        model.eval()
        policy = None
    elif args.mode in ['local', 'wo_co']:
        policy = LocalPolicy(config=config)
    elif args.mode == 'greedy':
        policy = GreedyPolicy(config=config)

    total_rewards = []
    total_delays = []
    total_energies = []
    total_completed_tasks = 0
    total_evaluated_tasks = 0

    print(f"=== Starting Evaluation ===")
    print(f"Mode: {args.mode} | Geometry: {sim_geom} | Mobility Aware: {use_mobility} | Priority Aware: {use_priority} | Episodes: {args.episodes}")

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        
        ep_reward = 0
        ep_delay = 0
        ep_energy = 0
        ep_completed = 0
        ep_tasks = 0
        
        while not done:
            if policy is not None:
                action = policy.select_action(obs)
            else:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    if args.mode == 'ddqn':
                        logits = model(obs_tensor)
                    else:
                        logits, _ = model(obs_tensor)
                mask = env.get_action_mask()
                mask_tensor = torch.BoolTensor(mask).unsqueeze(0)
                logits[~mask_tensor] = -1e9
                action = torch.argmax(logits, dim=-1).item()
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            ep_tasks += 1
            if 'delay' in info:
                ep_delay += info['delay']
                ep_energy += info['energy']
                curr_task = env.current_tasks[env.current_task_idx - 1] if env.current_task_idx > 0 else None
                if curr_task and info['delay'] <= curr_task.max_delay_d:
                    ep_completed += 1
                
        total_rewards.append(ep_reward)
        total_delays.append(ep_delay / max(ep_tasks, 1))
        total_energies.append(ep_energy / max(ep_tasks, 1))
        total_completed_tasks += ep_completed
        total_evaluated_tasks += ep_tasks
        
        comp_ratio_ep = (ep_completed / ep_tasks) if ep_tasks > 0 else 0.0
        print(f"Episode {episode+1:02d} | Reward: {ep_reward:6.2f} | Avg Delay: {total_delays[-1]:5.2f}s | Avg Energy: {total_energies[-1]:5.2f}J | Comp Ratio: {comp_ratio_ep * 100:.1f}%")

    overall_completion_ratio = (total_completed_tasks / total_evaluated_tasks) if total_evaluated_tasks > 0 else 0.0
    print("=" * 50)
    print(f"       FINAL EVALUATION REPORT ({args.mode.upper()})")
    print("=" * 50)
    print(f"Average Reward:           {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Delay (s):        {np.mean(total_delays):.2f} ± {np.std(total_delays):.2f}")
    print(f"Average Energy (J):       {np.mean(total_energies):.2f} ± {np.std(total_energies):.2f}")
    print(f"Task Completion Ratio:    {overall_completion_ratio * 100:.2f}%")
    print("=" * 50)
    
    if args.output_csv:
        import pandas as pd
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        df = pd.DataFrame({
            'episode': list(range(1, args.episodes + 1)),
            'mode': [args.mode] * args.episodes,
            'seed': [args.seed] * args.episodes,
            'reward': total_rewards,
            'delay': total_delays,
            'energy': total_energies
        })
        df.to_csv(args.output_csv, index=False)
        print(f"[INFO] Saved evaluation results to {args.output_csv}")

    env.close()

if __name__ == '__main__':
    evaluate()