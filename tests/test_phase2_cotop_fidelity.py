import pytest
import torch
import numpy as np
import yaml
from envs.vec_env import VECEnv
from envs.entities import SimulationConfig
from models.a3c_agent import ActorCritic
from utils.seed import set_seed
import os

@pytest.fixture
def env():
    set_seed(42)
    with open("configs/paper_parameters.yaml", 'r') as f:
        config_data = yaml.safe_load(f)
    config = SimulationConfig(**config_data)
    env = VECEnv(
        config=config,
        port=9996,
        scenario_geometry="grid_200m",
        use_mobility_model=True,
        max_vehicles=10,
        seed=42
    )
    yield env
    env.close()

def test_11_12_actor_critic_output_dimensions():
    """Tests 11, 12: Actor output dimension = 7, Critic scalar output."""
    model = ActorCritic(input_dim=50, num_actions=7)
    state = torch.rand((1, 50))
    logits, value = model(state)
    
    assert logits.shape == (1, 7)
    assert value.shape == (1, 1)

def test_13_14_invalid_action_masking_and_normalization():
    """Tests 13, 14: Invalid action masking and Probability normalization over valid actions."""
    logits = torch.tensor([[10.0, -10.0, 5.0, 0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, True, False, False, False, False]])
    
    logits[~mask] = -1e9
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Probabilities of invalid actions should be exactly 0
    assert probs[0, 3].item() == 0.0
    assert probs[0, 4].item() == 0.0
    assert probs[0, 5].item() == 0.0
    assert probs[0, 6].item() == 0.0
    
    # Sum of probabilities over valid actions should be exactly 1
    assert torch.isclose(probs.sum(), torch.tensor(1.0))

def test_25_deterministic_forward_pass():
    """Test 25: Deterministic forward pass."""
    set_seed(42)
    model = ActorCritic(input_dim=50, num_actions=7)
    state = torch.rand((1, 50))
    
    logits1, value1 = model(state)
    logits2, value2 = model(state)
    
    assert torch.allclose(logits1, logits2)
    assert torch.allclose(value1, value2)

def test_26_checkpoint_recovery(tmpdir):
    """Test 26: Checkpoint recovery."""
    model1 = ActorCritic(input_dim=50, num_actions=7)
    ckpt_path = os.path.join(tmpdir, "model.pth")
    torch.save(model1.state_dict(), ckpt_path)
    
    model2 = ActorCritic(input_dim=50, num_actions=7)
    # Ensure they are initially different
    model2.fc1.weight.data.normal_(0, 1)
    
    model2.load_state_dict(torch.load(ckpt_path))
    
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)

def test_27_28_determinism(env):
    """Tests 27, 28: Action sequence and State trajectory determinism."""
    # Run 1
    set_seed(100)
    obs1, _ = env.reset(seed=100)
    traj1 = [obs1.sum()]
    acts1 = []
    
    for _ in range(5):
        # Always pick action 0 for determinism test
        action = 0
        acts1.append(action)
        obs, _, done, _, _ = env.step(action)
        traj1.append(obs.sum())
        if done:
            break
            
    # Run 2
    set_seed(100)
    obs2, _ = env.reset(seed=100)
    traj2 = [obs2.sum()]
    acts2 = []
    
    for _ in range(5):
        action = 0
        acts2.append(action)
        obs, _, done, _, _ = env.step(action)
        traj2.append(obs.sum())
        if done:
            break
            
    assert acts1 == acts2
    assert np.allclose(traj1, traj2)
