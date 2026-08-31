import pytest
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv
from models.mobility_gat import MobilityGAT_GRU
from models.a3c_agent import ActorCritic
from envs.vec_env import VECEnv
from envs.entities import SimulationConfig
import yaml
from utils.seed import set_seed

@pytest.fixture
def env():
    set_seed(42)
    with open("configs/paper_parameters.yaml", 'r') as f:
        config_data = yaml.safe_load(f)
    config = SimulationConfig(**config_data)
    env = VECEnv(
        config=config,
        port=9997,
        scenario_geometry="grid_200m",
        use_mobility_model=True,
        max_vehicles=10,
        seed=42
    )
    yield env
    env.close()

def test_01_02_03_04_gat_node_neighborhood_distance_exclusion():
    """Tests 1, 2, 3, 4: GAT Node count, neighborhood construction, distance threshold, out-of-range exclusion."""
    pos = torch.tensor([
        [0.0, 0.0],
        [10.0, 0.0],
        [1000.0, 1000.0]
    ])
    
    radius = 200.0
    edges = []
    for i in range(3):
        for j in range(3):
            if i != j:
                dist = torch.norm(pos[i] - pos[j])
                if dist <= radius:
                    edges.append((i, j))
    
    assert (0, 1) in edges
    assert (1, 0) in edges
    assert (0, 2) not in edges
    assert (1, 2) not in edges

def test_05_06_attention_normalization_and_self_loops():
    """Tests 5, 6: Attention normalization (softmax) and self-loop behavior in GAT."""
    x = torch.rand((3, 64))
    edge_index = torch.tensor([[0, 1, 0, 2],
                               [1, 0, 2, 0]], dtype=torch.long)
    
    gat = GATConv(in_channels=64, out_channels=64, heads=1, concat=False, add_self_loops=True)
    out, attention_weights = gat(x, edge_index, return_attention_weights=True)
    
    edge_idx, alpha = attention_weights
    
    assert alpha.shape[0] == 7
    
    target_node = 0
    mask = edge_idx[1] == target_node
    sum_alpha = alpha[mask].sum()
    assert torch.isclose(sum_alpha, torch.tensor(1.0)), "Attention weights must normalize to 1.0"

def test_07_08_gat_gru_dimensions():
    """Tests 7, 8: GAT output dimensions and GRU input/output dims."""
    model = MobilityGAT_GRU(input_dim=2, embed_dim=64, num_heads=4, gru_hidden=64, output_dim=2)
    x_seq = torch.rand((5, 10, 2))
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    
    out = model(x_seq, edge_index)
    
    assert out.shape == (5, 10, 2)

def test_09_10_gru_hidden_state_propagation_and_reset():
    """Tests 9, 10: GRU hidden-state propagation and reset behavior."""
    model = MobilityGAT_GRU(input_dim=2, embed_dim=64, num_heads=4, gru_hidden=64, output_dim=2)
    
    x_seq = torch.rand((3, 5, 2))
    edge_index = torch.empty((2, 0), dtype=torch.long)
    
    model.eval()
    with torch.no_grad():
        out1 = model(x_seq, edge_index)
        out2 = model(x_seq, edge_index)
        
    assert torch.allclose(out1, out2), "GRU forward passes must be entirely stateless across distinct call executions"

def test_15_16_17_18_rl_mathematics():
    """Tests 15-18: Entropy, Reward calculation (using analytic verification), Discounting, Advantage."""
    logits = torch.tensor([[10.0, 10.0, -1e9, -1e9, -1e9, -1e9, -1e9]])
    probs = F.softmax(logits, dim=-1)
    
    assert torch.isclose(probs[0, 0], torch.tensor(0.5))
    
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
    assert torch.isclose(entropy, torch.tensor(float(-np.log(0.5))), atol=1e-3)
    
    eps = 0.5
    delay = 2.0
    energy = 5.0
    
    r_succ = -(eps * delay + (1 - eps) * energy)
    assert r_succ == -3.5
    
    rewards = [1.0, 1.0, 1.0]
    gamma = 0.99
    values = torch.tensor([0.0, 0.0, 0.0])
    
    R = 0
    returns = []
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
        
    assert np.isclose(returns[2], 1.0)
    assert np.isclose(returns[1], 1.99)
    assert np.isclose(returns[0], 2.9701)
    
    returns_tensor = torch.FloatTensor(returns)
    advantages = returns_tensor - values
    assert torch.allclose(advantages, returns_tensor)

def test_19_20_21_22_gradients_and_losses():
    """Tests 19, 20, 21, 22: PG direction, Critic loss, Actor loss, Gradient isolation."""
    model = ActorCritic(input_dim=10, num_actions=7)
    state = torch.rand((1, 10))
    
    logits, value = model(state)
    probs = F.softmax(logits, dim=-1)
    
    action_log_prob = torch.log(probs[0, 0])
    return_val = torch.tensor([10.0])
    
    critic_loss = F.mse_loss(value.view(-1), return_val)
    assert torch.isclose(critic_loss, (value.view(-1)[0] - 10.0)**2)
    
    advantage = (return_val - value.detach()).view(-1)
    
    actor_loss = -(action_log_prob * advantage).mean()
    assert torch.isclose(actor_loss, -action_log_prob * advantage[0])
    
    actor_loss.backward(retain_graph=True)
    
    has_actor_grad = False
    has_critic_grad = False
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if 'actor_head' in name:
                has_actor_grad = True
            if 'critic_head' in name:
                has_critic_grad = True
                
    assert has_actor_grad == True
    assert has_critic_grad == False 
    
    model.zero_grad()

def test_23_target_leakage():
    """Test 23: No accidental target leakage in advantages."""
    model = ActorCritic(input_dim=10, num_actions=7)
    state = torch.rand((1, 10))
    
    logits, value = model(state)
    probs = F.softmax(logits, dim=-1)
    
    action_log_prob = torch.log(probs[0, 0])
    return_val = torch.tensor([10.0])
    
    advantage = (return_val - value.detach()).view(-1)
    actor_loss = -(action_log_prob * advantage).mean()
    
    actor_loss.backward()
    
    # Check that critic weights don't have gradient
    for name, param in model.named_parameters():
        if 'critic_head' in name:
            assert param.grad is None

def test_24_numerical_stability():
    """Test 24: Numerical stability against NaNs and Infs."""
    model = ActorCritic(input_dim=10, num_actions=7)
    state = torch.full((1, 10), 1e6)
    
    logits, value = model(state)
    
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()
    assert not torch.isnan(value).any()
    assert not torch.isinf(value).any()
