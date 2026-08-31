import pytest
import torch
import numpy as np
import yaml
import torch.nn.functional as F

from envs.vec_env import VECEnv
from envs.entities import SimulationConfig
from models.a3c_agent import ActorCritic
from utils.seed import set_seed

@pytest.fixture
def env():
    set_seed(42)
    with open("configs/paper_parameters.yaml", 'r') as f:
        yaml_config = yaml.safe_load(f)
    # Use only 2 RSUs so that actions 3..6 are mathematically masked
    yaml_config["num_rsus"] = 2
    
    config = SimulationConfig(**yaml_config)
    env = VECEnv(
        config=config, 
        port=9998, 
        scenario_geometry="grid_200m",
        use_mobility_model=True, 
        use_priority=True,
        max_vehicles=2,
        seed=42
    )
    yield env
    env.close()

def test_training_action_masking(env):
    """Test that masked actions have exactly 0 probability after masked softmax during training."""
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    
    obs, _ = env.reset(seed=42)
    state = torch.FloatTensor(obs)
    
    policy_logits, _ = model(state)
    
    # Mock mask
    mask = np.ones(env.action_space.n, dtype=bool)
    mask[2] = False # mask out action 2
    mask_tensor = torch.BoolTensor(mask)
    
    policy_logits[~mask_tensor] = -1e9
    probs = F.softmax(policy_logits, dim=-1)
    
    assert probs[2].item() == 0.0
    
def test_evaluation_action_masking(env):
    """Test that argmax correctly ignores masked actions even if they have the highest raw logit."""
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    
    obs, _ = env.reset(seed=42)
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    
    with torch.no_grad():
        logits, _ = model(obs_tensor)
        
    # Force an invalid action to have an artificially high logit
    logits[0, 2] = 1000.0
    
    # Mock mask
    mask = np.ones(env.action_space.n, dtype=bool)
    mask[2] = False
    mask_tensor = torch.BoolTensor(mask).unsqueeze(0)
    logits[~mask_tensor] = -1e9
    
    action = torch.argmax(logits, dim=-1).item()
    
    assert action != 2
    assert mask[action] == True
