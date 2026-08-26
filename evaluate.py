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
                        choices=['cotop', 'local', 'greedy', 'wo_md', 'wo_tp', 'wo_co'], 
                        default='cotop')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default='configs/paper_parameters.yaml')
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)
    config = SimulationConfig(**yaml_config)
    
    # Configure ablation flags cleanly
    use_mobility = (args.mode != 'wo_md')
    use_priority = (args.mode != 'wo_tp')
    
    env = VECEnv(
        config=config, 
        port=9999, 
        use_mobility_model=use_mobility, 
        use_priority=use_priority,
        seed=args.seed
    )
    
    # Initialize Policies
    if args.mode in ['cotop', 'wo_md', 'wo_tp']:
        model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
        ckpt_path = 'results/checkpoints/a3c_agent.pth'
        if os.path.exists(ckpt_path):
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
                print(f"[INFO] Loaded trained CoTOP model from {ckpt_path}.")
            except Exception as e:
                print(f"[WARN] Error loading checkpoint ({e}). Using initialized weights.")
        else:
            print("[WARN] No checkpoint found. Evaluating with untrained agent.")
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
    print(f"Mode: {args.mode} | Mobility Aware: {use_mobility} | Priority Aware: {use_priority} | Episodes: {args.episodes}")

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
                    logits, _ = model(obs_tensor)
                action = torch.argmax(logits, dim=-1).item()
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            ep_tasks += 1
            if 'delay' in info:
                ep_delay += info['delay']
                ep_energy += info['energy']
                # Check if task completed within deadline (Section V-A Completion Ratio)
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
    
    env.close()

if __name__ == '__main__':
    evaluate()