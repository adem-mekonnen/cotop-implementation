import os
import threading
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from models.a3c_agent import ActorCritic
from envs.vec_env import VECEnv
from envs.entities import SimulationConfig

class A3CWorker(threading.Thread):
    def __init__(self, worker_id, global_model, optimizer, max_episodes, save_dir):
        super(A3CWorker, self).__init__()
        self.worker_id = worker_id
        self.global_model = global_model
        self.optimizer = optimizer
        self.max_episodes = max_episodes
        self.save_dir = save_dir
        
        # Local Environment for this worker thread
        config = SimulationConfig(bandwidth_B=10e6, noise_power_sigma2=1e-9)
        self.env = VECEnv(config=config, num_rsus=5, max_tasks=3)
        
        input_dim = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        
        # Local model specific to this thread
        self.local_model = ActorCritic(input_dim, num_actions)
        self.gamma = 0.99

    def run(self):
        for episode in range(self.max_episodes):
            # 1. Sync local model parameters with global model
            self.local_model.load_state_dict(self.global_model.state_dict())
            
            state = self.env.reset()
            state = torch.FloatTensor(state).unsqueeze(0)
            
            values, log_probs, rewards = [], [], []
            done = False
            
            # --- Algorithm 1: Task Offloading Training Loop ---
            while not done:
                # Forward pass
                policy_logits, value = self.local_model(state)
                probs = F.softmax(policy_logits, dim=-1)
                
                # Sample action from policy
                m = Categorical(probs)
                action = m.sample()
                
                # Step environment
                next_state, reward, done, _ = self.env.step(action.item())
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                
                # Store transitions
                values.append(value)
                log_probs.append(m.log_prob(action))
                rewards.append(reward)
                
                state = next_state
            
            # --- Calculate Returns and Loss ---
            R = 0
            returns = []
            for r in rewards[::-1]:
                R = r + self.gamma * R
                returns.insert(0, R)
                
            returns = torch.FloatTensor(returns)
            
            # Squeeze values to match returns shape if necessary. 
            # Check length to avoid dim errors on empty trajectories.
            if len(values) > 0:
                values = torch.cat(values).squeeze(-1)
                log_probs = torch.cat(log_probs)
                
                # Advantage estimation A(s,a) = R - V(s)
                advantages = returns - values.detach()
                
                actor_loss = -(log_probs * advantages).mean()
                critic_loss = F.mse_loss(values, returns)
                total_loss = actor_loss + 0.5 * critic_loss
                
                # Backpropagation onto local gradients
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Push local gradients to the global model
                for global_param, local_param in zip(self.global_model.parameters(), self.local_model.parameters()):
                    if local_param.grad is not None:
                        global_param._grad = local_param.grad
                    
                self.optimizer.step()
            
            # Log & Save Checkpoints every 50 episodes (using Worker 0 to avoid conflicts)
            if self.worker_id == 0 and (episode + 1) % 50 == 0:
                os.makedirs(self.save_dir, exist_ok=True)
                save_path = os.path.join(self.save_dir, f"cotop_model_ep_{episode+1}.pth")
                torch.save(self.global_model.state_dict(), save_path)
                print(f"[Worker 0] Episode {episode+1}/{self.max_episodes} - Saved checkpoint to {save_path}")

def train():
    # Avoid multithreading overhead in OpenMP since we are using Python threading
    os.environ['OMP_NUM_THREADS'] = '1'
    
    print("Initializing global environment and model...")
    config = SimulationConfig(bandwidth_B=10e6, noise_power_sigma2=1e-9)
    dummy_env = VECEnv(config=config)
    input_dim = dummy_env.observation_space.shape[0]
    num_actions = dummy_env.action_space.n
    
    # Create the global Actor-Critic model
    global_model = ActorCritic(input_dim, num_actions)
    global_model.share_memory() # Important for A3C
    
    # Global optimizer
    optimizer = optim.Adam(global_model.parameters(), lr=1e-3)
    
    num_workers = 4 # Number of parallel threads
    max_episodes = 500
    save_dir = "checkpoints"
    
    print(f"Starting {num_workers} A3C parallel workers using threading...")
    workers = []
    for i in range(num_workers):
        worker = A3CWorker(i, global_model, optimizer, max_episodes, save_dir)
        worker.start()
        workers.append(worker)
        
    for worker in workers:
        worker.join()
        
    print("Training Complete!")

if __name__ == "__main__":
    train()
