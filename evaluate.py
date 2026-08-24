import argparse
import torch
import numpy as np
import yaml

from envs.vec_env import VECEnv, get_euclidean_distance
from envs.entities import SimulationConfig
from models.a3c_agent import ActorCritic

def get_nearest_rsu(vehicle, rsus):
    nearest_idx = 0
    min_dist = float('inf')
    for i, rsu in enumerate(rsus):
        dist = get_euclidean_distance(vehicle.pos, rsu.location)
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i
    return nearest_idx

def get_greedy_rsu(rsus):
    greedy_idx = 0
    min_queue = float('inf')
    for i, rsu in enumerate(rsus):
        if rsu.queue_length < min_queue:
            min_queue = rsu.queue_length
            greedy_idx = i
    return greedy_idx

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['cotop', 'local', 'greedy'], default='cotop')
    parser.add_argument('--no_mobility', action='store_true', help='Ablation: Force low dwell time')
    parser.add_argument('--episodes', type=int, default=10)
    args = parser.parse_args()

    with open('configs/simulation.yaml', 'r') as f:
        yaml_config = yaml.safe_load(f)
    config = SimulationConfig(**yaml_config)
    
    use_mobility = not args.no_mobility
    env = VECEnv(config=config, port=9999)
    
    if args.mode == 'cotop':
        model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
        try:
            model.load_state_dict(torch.load('results/checkpoints/a3c_agent.pth', map_location='cpu'))
            print("Loaded trained CoTOP model.")
        except Exception as e:
            print(f"Could not load CoTOP model ({e}). Evaluating with untrained weights.")
        model.eval()

    total_rewards = []
    total_delays = []
    total_energies = []

    print(f"Starting Evaluation - Mode: {args.mode}, Mobility: {use_mobility}")

    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        
        ep_reward = 0
        ep_delay = 0
        ep_energy = 0
        
        while not done:
            # Ablation Logic: Override dwell time prediction
            if args.no_mobility:
                env.current_vehicle.dwell_time_T_stay = 2.0
                
            if args.mode == 'cotop':
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = model(obs_tensor)
                # Greedy selection during evaluation instead of sampling
                action = torch.argmax(logits, dim=-1).item()
                obs, reward, terminated, truncated, info = env.step(action)
                
            elif args.mode == 'local':
                action = get_nearest_rsu(env.current_vehicle, env.rsus)
                
                # Local baseline enforces Standalone Case 1 only.
                # Setting dwell time to infinity ensures Case 2 (collaboration) is never triggered
                # inside the env's step logic for this mode.
                env.current_vehicle.dwell_time_T_stay = float('inf') 
                obs, reward, terminated, truncated, info = env.step(action)
                
            elif args.mode == 'greedy':
                action = get_greedy_rsu(env.rsus)
                obs, reward, terminated, truncated, info = env.step(action)
                
            done = terminated or truncated
            ep_reward += reward
            if 'delay' in info:
                ep_delay += info['delay']
                ep_energy += info['energy']
                
        total_rewards.append(ep_reward)
        total_delays.append(ep_delay)
        total_energies.append(ep_energy)
        
        print(f"Episode {episode+1} | Reward: {ep_reward:.2f} | Delay: {ep_delay:.2f} | Energy: {ep_energy:.2f}")

    print("-" * 40)
    print(f"EVALUATION RESULTS ({args.mode.upper()})")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Average Delay:  {np.mean(total_delays):.2f}")
    print(f"Average Energy: {np.mean(total_energies):.2f}")
    print("-" * 40)
    
    env.close()

if __name__ == '__main__':
    evaluate()