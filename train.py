import os
import threading
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import yaml

from models.a3c_agent import ActorCritic
from envs.vec_env import VECEnv
from envs.entities import SimulationConfig

class A3CWorker(threading.Thread):
    def __init__(self, worker_id, global_model, optimizer, config, save_dir, mobility_model_path):
        super(A3CWorker, self).__init__()
        self.worker_id = worker_id
        self.global_model = global_model
        self.optimizer = optimizer
        self.save_dir = save_dir
        
        # Initialize local environment with full config
        self.env = VECEnv(config=config)
        
        # Dynamic Dims: Get exactly what the environment provides
        input_dim = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        
        self.local_model = ActorCritic(input_dim, num_actions)
        self.gamma = 0.99
        self.max_episodes = 1000 # Paper usually trains for ~1000 eps

    def run(self):
        for episode in range(self.max_episodes):
            # Sync with global model
            self.local_model.load_state_dict(self.global_model.state_dict())
            
            # Gymnasium API: reset returns (state, info)
            state, _ = self.env.reset()
            state = torch.FloatTensor(state)
            
            values, log_probs, rewards = [], [], []
            done = False
            
            while not done:
                policy_logits, value = self.local_model(state)
                probs = F.softmax(policy_logits, dim=-1)
                
                m = Categorical(probs)
                action = m.sample()
                
                # Gymnasium API: step returns (next_state, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, info = self.env.step(action.item())
                done = terminated or truncated
                
                values.append(value)
                log_probs.append(m.log_prob(action))
                rewards.append(reward)
                
                state = torch.FloatTensor(next_state)
            
            # --- Returns and Loss Calculation (Algorithm 1, Lines 15-20) ---
            R = 0
            returns = []
            for r in rewards[::-1]:
                R = r + self.gamma * R
                returns.insert(0, R)
                
            returns = torch.FloatTensor(returns)
            if len(values) > 0:
                values = torch.cat(values).squeeze()
                log_probs = torch.cat(log_probs)
                
                advantages = returns - values.detach()
                actor_loss = -(log_probs * advantages).mean()
                critic_loss = F.mse_loss(values, returns)
                total_loss = actor_loss + 0.5 * critic_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Manual Gradient Sharing for A3C
                for global_param, local_param in zip(self.global_model.parameters(), self.local_model.parameters()):
                    global_param._grad = local_param.grad
                self.optimizer.step()
            
            if self.worker_id == 0 and (episode + 1) % 50 == 0:
                print(f"Episode {episode+1} | Reward: {sum(rewards):.2f}")
                torch.save(self.global_model.state_dict(), os.path.join(self.save_dir, "a3c_agent.pth"))

def train():
    # 1. Load Simulation Config (Verified Table III)
    with open("configs/simulation.yaml", 'r') as f:
        config_data = yaml.safe_load(f)
    config = SimulationConfig(**config_data)
    
    os.makedirs("results/checkpoints", exist_ok=True)
    
    # 2. Get Dynamic Dims from a dummy env to prevent "Size Mismatch"
    temp_env = VECEnv(config=config)
    input_dim = temp_env.observation_space.shape[0]
    num_actions = temp_env.action_space.n
    temp_env.close()
    
    print(f"Starting A3C with State Dim: {input_dim}, Action Dim: {num_actions}")

    global_model = ActorCritic(input_dim, num_actions)
    global_model.share_memory()
    optimizer = optim.Adam(global_model.parameters(), lr=0.0002)
    
    num_workers = 4 
    workers = []
    for i in range(num_workers):
        worker = A3CWorker(i, global_model, optimizer, config, "results/checkpoints", "results/checkpoints/mobility_model.pth")
        worker.start()
        workers.append(worker)
        
    for worker in workers:
        worker.join()

if __name__ == "__main__":
    train()