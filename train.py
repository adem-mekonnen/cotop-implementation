import os
import time
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical
import yaml
import numpy as np

from models.a3c_agent import ActorCritic
from envs.vec_env import VECEnv
from envs.entities import SimulationConfig
from utils.seed import set_seed

class SharedAdam(optim.Adam):
    """
    Implements a SharedAdam optimizer for A3C parallel training.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


def worker_process(
    worker_id,
    global_model,
    optimizer,
    config,
    max_episodes,
    save_dir,
    mobility_model_path,
    seed_base,
    scenario_geometry="corridor_2400m",
    use_mobility_model=True,
    use_priority=True,
    priority_mode="paper_literal",
    coverage_mode="completion_position",
    spatial_graph_radius=200.0,
    max_vehicles=10
):
    worker_seed = seed_base + worker_id
    set_seed(worker_seed)
    
    # Canonical mapping
    sim_geom = "grid_200m" if scenario_geometry in ["grid_200m", "urban_manhattan"] else "corridor_2400m"
    port = 8813 + (worker_id * 2)
    
    env = VECEnv(
        config=config,
        port=port,
        scenario_geometry=sim_geom,
        use_mobility_model=use_mobility_model,
        use_priority=use_priority,
        priority_mode=priority_mode,
        coverage_mode=coverage_mode,
        spatial_graph_radius=spatial_graph_radius,
        max_vehicles=max_vehicles,
        seed=worker_seed
    )
    
    input_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    local_model = ActorCritic(input_dim, num_actions)
    gamma = 0.99

    time.sleep(worker_id * 1.5)
    for episode in range(max_episodes):
        local_model.load_state_dict(global_model.state_dict())
        
        state, _ = env.reset(seed=worker_seed + episode)
        state = torch.FloatTensor(state)
        
        values, log_probs, rewards = [], [], []
        done = False
        
        while not done:
            policy_logits, value = local_model(state)
            probs = F.softmax(policy_logits, dim=-1)
            
            m = Categorical(probs)
            action = m.sample()
            
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            values.append(value)
            log_probs.append(m.log_prob(action))
            rewards.append(reward)
            
            state = torch.FloatTensor(next_state)
        
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.FloatTensor(returns)
        if len(values) > 0:
            values = torch.stack(values).view(-1)
            log_probs = torch.stack(log_probs).view(-1)
            
            advantages = returns - values.detach()
            actor_loss = -(log_probs * advantages).mean()
            critic_loss = F.mse_loss(values, returns)
            
            probs_all = F.softmax(local_model(state)[0].detach(), dim=-1)
            entropy = -(probs_all * (probs_all + 1e-8).log()).sum(dim=-1).mean()
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            total_loss.backward()
            
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                if local_param.grad is not None:
                    global_param._grad = local_param.grad
            optimizer.step()
        
        if worker_id == 0 and (episode + 1) % 10 == 0:
            print(f"[Worker 0] Episode {episode+1:03d}/{max_episodes} | Total Reward: {sum(rewards):6.2f}")
            torch.save(global_model.state_dict(), os.path.join(save_dir, "a3c_agent.pth"))

    try:
        env.close()
    except Exception:
        pass


def train(args):
    set_seed(args.seed)
    
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
        
    if args.workload is not None:
        config_data["num_tasks_per_vehicle_range"] = [args.workload, args.workload]
        
    config = SimulationConfig(**config_data)
    
    sim_geom = "grid_200m" if args.scenario_geometry in ["grid_200m", "urban_manhattan"] else "corridor_2400m"
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    temp_env = VECEnv(
        config=config,
        scenario_geometry=sim_geom,
        use_mobility_model=args.use_mobility_model,
        use_priority=args.use_priority,
        priority_mode=args.priority_mode,
        coverage_mode=args.coverage_mode,
        spatial_graph_radius=args.spatial_graph_radius,
        max_vehicles=args.max_vehicles,
        seed=args.seed
    )
    input_dim = temp_env.observation_space.shape[0]
    num_actions = temp_env.action_space.n
    temp_env.close()
    
    print(f"=== Starting A3C Training ===")
    print(f"Geometry: {sim_geom} | State Dim: {input_dim} | Action Dim: {num_actions} | Workers: {args.workers} | Max Episodes: {args.episodes}")

    global_model = ActorCritic(input_dim, num_actions)
    global_model.share_memory()
    optimizer = SharedAdam(global_model.parameters(), lr=args.lr)
    
    processes = []
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    for i in range(args.workers):
        p = mp.Process(
            target=worker_process, 
            args=(
                i,
                global_model,
                optimizer,
                config,
                args.episodes,
                args.save_dir,
                "results/checkpoints/mobility_model.pth",
                args.seed,
                args.scenario_geometry,
                args.use_mobility_model,
                args.use_priority,
                args.priority_mode,
                args.coverage_mode,
                args.spatial_graph_radius,
                args.max_vehicles
            )
        )
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

    ckpt_path = os.path.join(args.save_dir, "a3c_agent.pth")
    torch.save(global_model.state_dict(), ckpt_path)
    print(f"[SUCCESS] Final A3C model saved to {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/paper_parameters.yaml")
    parser.add_argument("--save_dir", type=str, default="results/checkpoints")
    parser.add_argument("--scenario_geometry", type=str, default="corridor_2400m", choices=["corridor_2400m", "grid_200m", "linear_corridor", "urban_manhattan"])
    parser.add_argument("--use_mobility_model", action="store_true", default=True)
    parser.add_argument("--no_mobility_model", dest="use_mobility_model", action="store_false")
    parser.add_argument("--use_priority", action="store_true", default=True)
    parser.add_argument("--no_priority", dest="use_priority", action="store_false")
    parser.add_argument("--priority_mode", type=str, default="paper_literal", choices=["paper_literal", "normalized_candidate"])
    parser.add_argument("--coverage_mode", type=str, default="completion_position", choices=["completion_position", "continuous_required_rsus"])
    parser.add_argument("--spatial_graph_radius", type=float, default=200.0)
    parser.add_argument("--max_vehicles", type=int, default=10)
    parser.add_argument("--workload", type=int, default=None, help="Tasks per vehicle override")
    args = parser.parse_args()
    
    train(args)