import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    """
    A3C Agent for CoTOP Task Offloading.
    Implements a 3 fully-connected layer architecture (Sec IV-E-1).
    """
    def __init__(self, input_dim: int, num_actions: int, hidden_size: int = 128):
        super(ActorCritic, self).__init__()
        
        # 1st Layer: Shared representation layer
        self.fc1 = nn.Linear(input_dim, hidden_size)
        
        # 2nd Layer: Shared hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # 3rd Layer: Branched into Actor and Critic heads
        # Actor Head: Outputs policy logits over possible RSUs to offload to
        self.actor_head = nn.Linear(hidden_size, num_actions)
        
        # Critic Head: Outputs State-Value estimation V(s)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, state):
        """
        Forward pass through the Actor-Critic networks.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Policy logits (to be passed to softmax for action probabilities)
        policy_logits = self.actor_head(x)
        
        # Value estimation
        state_value = self.critic_head(x)
        
        return policy_logits, state_value
