import argparse
import torch
import numpy as np
import torch.nn.functional as F

from envs.vec_env import VECEnv
from envs.entities import SimulationConfig
from models.a3c_agent import ActorCritic
from models.mobility_gat import MobilityGAT_GRU

def evaluate(args):
    print(f"Starting Evaluation for {args.episodes} episodes...")
    
    # 1. Initialize Environment
    config = SimulationConfig(bandwidth_B=10e6, noise_power_sigma2=1e-9)
    env = VECEnv(config=config, num_rsus=5, max_tasks=3)
    
    input_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    # 2. Load Models
    agent_model = ActorCritic(input_dim, num_actions)
    if args.a3c_model:
        try:
            agent_model.load_state_dict(torch.load(args.a3c_model))
            print(f"Loaded A3C model from {args.a3c_model}")
        except FileNotFoundError:
            print("A3C model not found. Using untrained weights.")
    agent_model.eval()
            
    # Load Mobility Model if not disabled via ablation flag
    if not args.no_mobility:
        gat_model = MobilityGAT_GRU(input_dim=2, embed_dim=64, num_heads=4, gru_hidden=64, output_dim=2)
        if args.gat_model:
            try:
                gat_model.load_state_dict(torch.load(args.gat_model))
                print(f"Loaded Mobility GAT model from {args.gat_model}")
            except FileNotFoundError:
                print("Mobility GAT model not found. Using untrained weights.")
        gat_model.eval()
    else:
        print("Ablation Enabled: Mobility Prediction Disabled (--no_mobility)")
        
    if args.no_priority:
        print("Ablation Enabled: Task Priority Sorting Disabled (--no_priority)")
        
    # 3. Metrics Tracking
    total_delay = 0.0
    total_energy = 0.0
    completed_tasks = 0
    total_tasks = 0
    
    # 4. Evaluation Loop
    with torch.no_grad():
        for ep in range(args.episodes):
            state = env.reset()
            done = False
            
            ep_delay = 0.0
            ep_energy = 0.0
            ep_completed = 0
            ep_tasks = 0
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # In evaluation, we take the greedy action (highest probability)
                policy_logits, _ = agent_model(state_tensor)
                probs = F.softmax(policy_logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item()
                
                next_state, reward, done, info = env.step(action)
                
                if info:
                    ep_delay += info.get("delay", 0)
                    ep_energy += info.get("energy", 0)
                    # Task is completed if it didn't miss the deadline
                    if not info.get("missed_deadline", True):
                        ep_completed += 1
                    ep_tasks += 1
                    
                state = next_state
                
            total_delay += ep_delay
            total_energy += ep_energy
            completed_tasks += ep_completed
            total_tasks += ep_tasks
            
    # 5. Calculate Final Metrics
    if total_tasks > 0:
        avg_delay = total_delay / total_tasks
        avg_energy = total_energy / total_tasks
        completion_ratio = completed_tasks / total_tasks
    else:
        avg_delay, avg_energy, completion_ratio = 0, 0, 0
        
    # Print formatted tables matching the paper's representation
    print("\n" + "="*45)
    print(" EVALUATION RESULTS ")
    print("="*45)
    print(f"Total Test Episodes : {args.episodes}")
    print(f"Average Task Delay  : {avg_delay:.4f} s (Table IV)")
    print(f"Completion Ratio    : {completion_ratio*100:.2f}% (Table V)")
    print(f"Average Energy      : {avg_energy:.4f} J")
    print("="*45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CoTOP DRL Agent")
    parser.add_argument("--a3c_model", type=str, default="", help="Path to trained A3C model (e.g., checkpoints/cotop_model_ep_500.pth)")
    parser.add_argument("--gat_model", type=str, default="", help="Path to trained Mobility GAT model")
    parser.add_argument("--episodes", type=int, default=50, help="Number of test episodes")
    
    # Ablation Study Flags (Table VI)
    parser.add_argument("--no_mobility", action="store_true", help="Disable the Mobility Prediction GAT-GRU model")
    parser.add_argument("--no_priority", action="store_true", help="Disable Equation 23 Task Priority queue")
    
    args = parser.parse_args()
    evaluate(args)
