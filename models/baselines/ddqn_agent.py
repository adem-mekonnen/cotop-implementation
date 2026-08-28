import random
from collections import deque
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    3 fully-connected layer Q-Network for DDQN Baseline.
    Architecture: 114 (State) -> 128 -> 128 -> 128 -> 7 (Actions) with ReLU.
    """
    def __init__(self, input_dim: int = 114, num_actions: int = 7, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing Q-values for all discrete actions.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.fc_out(x)
        return q_values


class ReplayBuffer:
    """
    Bounded FIFO experience replay buffer.
    Capacity: 10,000 transitions.
    """
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: Optional[np.ndarray] = None
    ) -> None:
        """
        Append transition (s, a, r, s', d, next_mask) to buffer.
        """
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
            np.array(next_action_mask, dtype=bool) if next_action_mask is not None else None
        ))

    def sample(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample a random minibatch of transitions.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

        if next_masks[0] is not None:
            next_masks_t = torch.tensor(np.array(next_masks), dtype=torch.bool, device=device)
        else:
            next_masks_t = None

        return states_t, actions_t, rewards_t, next_states_t, dones_t, next_masks_t

    def __len__(self) -> int:
        return len(self.buffer)


class DDQNAgent:
    """
    Audited Double Deep Q-Network (DDQN) Baseline Agent.
    Conforms to Reference [34] (Zhai et al., IEEE TVT 2024) and Phase 2 Specifications.
    
    Decoupled Target Construction:
      y_t = r_t + gamma * (1 - d_t) * Q_target(s_{t+1}, argmax_{a'} Q_online(s_{t+1}, a'))
    """
    def __init__(
        self,
        input_dim: int = 114,
        num_actions: int = 7,
        hidden_dim: int = 128,
        learning_rate: float = 0.0002,
        gamma: float = 0.99,
        replay_capacity: int = 10000,
        batch_size: int = 64,
        target_update_frequency: int = 100,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 200,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_capacity = replay_capacity
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.epsilon = epsilon_start

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Neural Networks
        self.online_net = QNetwork(input_dim, num_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(input_dim, num_actions, hidden_dim).to(self.device)
        
        # Initialize target network to match online network
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        # Freeze target network parameters (no gradient calculation)
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.target_net.eval()

        # 2. Optimizer and Loss Function
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction='mean', beta=1.0)

        # 3. Experience Replay
        self.memory = ReplayBuffer(capacity=self.replay_capacity)

        # 4. State Counters
        self.train_step_count = 0
        self.episode_count = 0

    def compute_epsilon(self, episode: Optional[int] = None) -> float:
        """
        Compute linear epsilon decay value at a given episode.
        Decays from epsilon_start (1.0) to epsilon_end (0.05) over epsilon_decay_episodes (200).
        """
        ep = self.episode_count if episode is None else episode
        if ep >= self.epsilon_decay_episodes:
            return float(self.epsilon_end)
        decay_step = (self.epsilon_start - self.epsilon_end) / float(self.epsilon_decay_episodes)
        eps = self.epsilon_start - (decay_step * ep)
        return float(np.clip(eps, self.epsilon_end, self.epsilon_start))

    def set_episode(self, episode: int) -> None:
        """
        Set current episode index and update epsilon value.
        """
        self.episode_count = int(episode)
        self.epsilon = self.compute_epsilon(self.episode_count)

    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action_mask: Optional[Union[np.ndarray, torch.Tensor, List[bool]]] = None,
        deterministic: bool = False
    ) -> int:
        """
        Select discrete action using epsilon-greedy (training) or pure greedy (evaluation).
        Strictly respects action feasibility mask if provided.
        """
        if action_mask is not None:
            if isinstance(action_mask, torch.Tensor):
                mask_np = action_mask.cpu().numpy().astype(bool)
            elif isinstance(action_mask, list):
                mask_np = np.array(action_mask, dtype=bool)
            else:
                mask_np = np.asarray(action_mask, dtype=bool)
            valid_actions = np.where(mask_np)[0]
            if len(valid_actions) == 0:
                # Fallback if no valid action (e.g. edge case): all actions allowed
                valid_actions = np.arange(self.num_actions)
        else:
            valid_actions = np.arange(self.num_actions)

        # Exploration (training mode only)
        if not deterministic and random.random() < self.epsilon:
            return int(random.choice(valid_actions))

        # Greedy action selection via online network
        self.online_net.eval()
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            elif isinstance(state, torch.Tensor):
                state_t = state.to(self.device)
                if state_t.dim() == 1:
                    state_t = state_t.unsqueeze(0)
            else:
                state_t = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)

            q_values = self.online_net(state_t).squeeze(0)  # Shape (num_actions,)

            # Apply action mask by penalizing invalid actions with -inf
            if action_mask is not None:
                mask_t = torch.tensor(mask_np, dtype=torch.bool, device=self.device)
                q_values = torch.where(mask_t, q_values, torch.tensor(-1e9, device=self.device))

            action = torch.argmax(q_values).item()

        return int(action)

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: Optional[np.ndarray] = None
    ) -> None:
        """
        Store a transition in the replay buffer.
        """
        self.memory.push(state, action, reward, next_state, done, next_action_mask)

    def update(self) -> Optional[float]:
        """
        Perform a single optimization step using a sampled minibatch.
        Implements Decoupled Double Q-learning target construction:
          y_t = r_t + gamma * (1 - d_t) * Q_target(s_{t+1}, argmax_{a'} Q_online(s_{t+1}, a'))
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample minibatch
        states, actions, rewards, next_states, dones, next_masks = self.memory.sample(
            self.batch_size, device=self.device
        )

        self.online_net.train()

        # 1. Compute Q(s, a; theta) using online network
        q_values = self.online_net(states)  # Shape (batch_size, num_actions)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # Shape (batch_size,)

        # 2. Compute DDQN Target y_t using decoupled action selection & target evaluation
        with torch.no_grad():
            # Online network selects best action a* for next state s_{t+1}
            next_online_q = self.online_net(next_states)  # Shape (batch_size, num_actions)
            if next_masks is not None:
                next_online_q = torch.where(next_masks, next_online_q, torch.tensor(-1e9, device=self.device))
            best_next_actions = torch.argmax(next_online_q, dim=1, keepdim=True)  # Shape (batch_size, 1)

            # Target network evaluates Q(s_{t+1}, a*; theta^-)
            next_target_q = self.target_net(next_states)  # Shape (batch_size, num_actions)
            next_q_values = next_target_q.gather(1, best_next_actions).squeeze(1)  # Shape (batch_size,)

            # Bellman target with terminal transition masking
            expected_state_action_values = rewards + self.gamma * (1.0 - dones) * next_q_values

        # 3. Compute Smooth L1 (Huber) Loss
        loss = self.loss_fn(state_action_values, expected_state_action_values)

        # 4. Backpropagate and update online network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 5. Increment training step counter and check target network synchronization
        self.train_step_count += 1
        if self.train_step_count % self.target_update_frequency == 0:
            self.sync_target_network()

        return float(loss.item())

    def sync_target_network(self) -> None:
        """
        Synchronize target network weights to match online network weights (theta^- <- theta).
        """
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save_checkpoint(self, filepath: str, extra_metadata: Optional[Dict] = None) -> None:
        """
        Serialize exact training state to checkpoint file.
        """
        checkpoint = {
            "online_net_state_dict": self.online_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_step_count": self.train_step_count,
            "episode_count": self.episode_count,
            "epsilon": self.epsilon,
            "config": {
                "input_dim": self.input_dim,
                "num_actions": self.num_actions,
                "hidden_dim": self.hidden_dim,
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "replay_capacity": self.replay_capacity,
                "batch_size": self.batch_size,
                "target_update_frequency": self.target_update_frequency,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay_episodes": self.epsilon_decay_episodes,
            },
            "extra_metadata": extra_metadata or {}
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """
        Deserialize and restore exact training state from checkpoint file.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_step_count = checkpoint["train_step_count"]
        self.episode_count = checkpoint["episode_count"]
        self.epsilon = checkpoint["epsilon"]
        
        # Ensure target network parameters remain frozen
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.target_net.eval()
