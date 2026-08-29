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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.fc_out(x)
        return q_values


class ReplayBuffer:
    """
    High-performance bounded FIFO experience replay buffer with preallocated memory.
    Capacity: 10,000 transitions.
    """
    def __init__(self, capacity: int = 10000, state_dim: int = 114, num_actions: int = 7):
        self.capacity = capacity
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.masks = np.ones((capacity, num_actions), dtype=bool)
        self.has_masks = np.zeros(capacity, dtype=bool)
        
        self.ptr = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: Optional[np.ndarray] = None
    ) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = 1.0 if done else 0.0
        
        if next_action_mask is not None:
            self.masks[self.ptr] = next_action_mask
            self.has_masks[self.ptr] = True
        else:
            self.has_masks[self.ptr] = False
            
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        indices = np.random.randint(0, self.size, size=batch_size)
        
        states_t = torch.from_numpy(self.states[indices]).float().to(device)
        actions_t = torch.from_numpy(self.actions[indices]).long().to(device)
        rewards_t = torch.from_numpy(self.rewards[indices]).float().to(device)
        next_states_t = torch.from_numpy(self.next_states[indices]).float().to(device)
        dones_t = torch.from_numpy(self.dones[indices]).float().to(device)
        
        if self.has_masks[indices[0]]:
            next_masks_t = torch.from_numpy(self.masks[indices]).bool().to(device)
        else:
            next_masks_t = None

        return states_t, actions_t, rewards_t, next_states_t, dones_t, next_masks_t

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.size:
            raise IndexError("ReplayBuffer index out of range")
        if self.size < self.capacity:
            real_idx = idx
        else:
            real_idx = (self.ptr + idx) % self.capacity
        return (
            self.states[real_idx],
            self.actions[real_idx],
            self.rewards[real_idx],
            self.next_states[real_idx],
            self.dones[real_idx],
            self.masks[real_idx] if self.has_masks[real_idx] else None
        )

    @property
    def buffer(self):
        return self

    def __len__(self) -> int:
        return self.size


class DDQNAgent:
    """
    Audited Double Deep Q-Network (DDQN) Baseline Agent.
    Conforms to Reference [34] (Zhai et al., IEEE TVT 2024) and Phase 2 Specifications.
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

        self.online_net = QNetwork(input_dim, num_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(input_dim, num_actions, hidden_dim).to(self.device)
        
        self.target_net.load_state_dict(self.online_net.state_dict())
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction='mean', beta=1.0)

        self.memory = ReplayBuffer(
            capacity=self.replay_capacity,
            state_dim=self.input_dim,
            num_actions=self.num_actions
        )

        self.train_step_count = 0
        self.episode_count = 0

    def compute_epsilon(self, episode: Optional[int] = None) -> float:
        ep = self.episode_count if episode is None else episode
        if ep >= self.epsilon_decay_episodes:
            return float(self.epsilon_end)
        decay_step = (self.epsilon_start - self.epsilon_end) / float(self.epsilon_decay_episodes)
        eps = self.epsilon_start - (decay_step * ep)
        return float(np.clip(eps, self.epsilon_end, self.epsilon_start))

    def set_episode(self, episode: int) -> None:
        self.episode_count = int(episode)
        self.epsilon = self.compute_epsilon(self.episode_count)

    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action_mask: Optional[Union[np.ndarray, torch.Tensor, List[bool]]] = None,
        deterministic: bool = False
    ) -> int:
        if action_mask is not None:
            if isinstance(action_mask, torch.Tensor):
                mask_np = action_mask.cpu().numpy().astype(bool)
            elif isinstance(action_mask, list):
                mask_np = np.array(action_mask, dtype=bool)
            else:
                mask_np = np.asarray(action_mask, dtype=bool)
            valid_actions = np.where(mask_np)[0]
            if len(valid_actions) == 0:
                valid_actions = np.arange(self.num_actions)
        else:
            valid_actions = np.arange(self.num_actions)

        if not deterministic and random.random() < self.epsilon:
            return int(random.choice(valid_actions))

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

            q_values = self.online_net(state_t).squeeze(0)

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
        self.memory.push(state, action, reward, next_state, done, next_action_mask)

    def update(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, next_masks = self.memory.sample(
            self.batch_size, device=self.device
        )

        self.online_net.train()

        q_values = self.online_net(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_online_q = self.online_net(next_states)
            if next_masks is not None:
                next_online_q = torch.where(next_masks, next_online_q, torch.tensor(-1e9, device=self.device))
            best_next_actions = torch.argmax(next_online_q, dim=1, keepdim=True)

            next_target_q = self.target_net(next_states)
            next_q_values = next_target_q.gather(1, best_next_actions).squeeze(1)

            expected_state_action_values = rewards + self.gamma * (1.0 - dones) * next_q_values

        loss = self.loss_fn(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_frequency == 0:
            self.sync_target_network()

        return float(loss.item())

    def sync_target_network(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save_checkpoint(self, filepath: str, extra_metadata: Optional[Dict] = None) -> None:
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
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint["online_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_step_count = checkpoint["train_step_count"]
        self.episode_count = checkpoint["episode_count"]
        self.epsilon = checkpoint["epsilon"]
        
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.target_net.eval()
