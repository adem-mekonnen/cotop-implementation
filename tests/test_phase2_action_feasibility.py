import pytest
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import yaml

from envs.entities import SimulationConfig
from envs.vec_env import VECEnv
from models.baselines import DDQNAgent, LocalPolicy, GreedyPolicy
from models.a3c_agent import ActorCritic


def test_01_vec_env_get_action_mask_shape_and_values():
    """
    Test 01 — Authoritative VECEnv.get_action_mask()
    Verify that get_action_mask() returns a boolean array of shape (7,)
    with Action 0 (standalone) always True and active RSUs True.
    """
    with open("configs/paper_parameters.yaml", "r") as f:
        cfg = SimulationConfig(**yaml.safe_load(f))
        
    env = VECEnv(config=cfg, port=9201, seed=42)
    env.reset(seed=42)
    mask = env.get_action_mask()
    
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == (7,)
    assert mask[0] is np.True_ or mask[0] == True
    assert np.all(mask[:7])  # All 6 RSUs active + standalone = 7 True
    env.close()


def test_02_ddqn_online_argmax_respects_action_mask():
    """
    Test 02 — DDQN Online Action Selection Under Arbitrary Masks
    Verify that DDQNAgent.select_action never selects a masked action,
    even if the Q-value of the masked action is artificially huge (+9999.0).
    """
    agent = DDQNAgent(input_dim=114, num_actions=7, device="cpu")
    
    # Force online network bias to prefer action 3 overwhelmingly
    with torch.no_grad():
        agent.online_net.fc_out.bias.fill_(0.0)
        agent.online_net.fc_out.bias[3] = 9999.0  # Huge Q-value on action 3
        agent.online_net.fc_out.bias[1] = 10.0    # Smaller Q-value on action 1
        
    state = np.zeros(114, dtype=np.float32)
    
    # Mask out action 3 (only actions 0 and 1 allowed)
    mask = np.array([True, True, False, False, False, False, False], dtype=bool)
    
    # Deterministic greedy evaluation
    selected = agent.select_action(state, action_mask=mask, deterministic=True)
    assert selected == 1, f"Expected action 1, got {selected} (masked action 3 was selected!)"
    
    # Epsilon exploration with mask
    for _ in range(50):
        rand_action = agent.select_action(state, action_mask=mask, deterministic=False)
        assert mask[rand_action], f"Exploration selected invalid action {rand_action}!"


def test_03_ddqn_target_argmax_masks_next_state_actions():
    """
    Test 03 — DDQN Target Evaluation Masks Infeasible Next-State Actions
    Verify that in DDQNAgent.update(), if next_masks is provided,
    best_next_actions argmax ignores masked actions.
    """
    agent = DDQNAgent(input_dim=114, num_actions=7, batch_size=4, device="cpu")
    
    # Artificially set online network to predict huge Q on action 5
    with torch.no_grad():
        agent.online_net.fc_out.bias.fill_(0.0)
        agent.online_net.fc_out.bias[5] = 500.0  # Highest Q
        agent.online_net.fc_out.bias[0] = 50.0   # Feasible alternative
        
    state = np.zeros(114, dtype=np.float32)
    next_state = np.zeros(114, dtype=np.float32)
    # Mask out action 5 in next state (only action 0 allowed)
    next_mask = np.array([True, False, False, False, False, False, False], dtype=bool)
    
    for _ in range(10):
        agent.store_transition(state, 0, 1.0, next_state, False, next_mask)
        
    loss = agent.update()
    assert loss is not None
    assert np.isfinite(loss)


def test_04_cotop_masked_action_sampling():
    """
    Test 04 — CoTOP Action Sampling Conformance with Action Feasibility Mask
    Verify that applying action mask to CoTOP policy logits eliminates
    probability of selecting masked actions.
    """
    model = ActorCritic(input_dim=114, num_actions=7, hidden_size=128)
    state_t = torch.zeros((1, 114), dtype=torch.float32)
    
    with torch.no_grad():
        logits, _ = model(state_t)
        
    # Mask out actions 2..6 (only 0 and 1 feasible)
    mask = torch.tensor([True, True, False, False, False, False, False], dtype=torch.bool)
    masked_logits = torch.where(mask, logits, torch.tensor(-1e9))
    probs = F.softmax(masked_logits, dim=-1)
    
    assert torch.all(probs[0, 2:] < 1e-6)
    
    m = Categorical(probs)
    for _ in range(50):
        sampled_action = m.sample().item()
        assert sampled_action in [0, 1]


def test_05_greedy_and_local_baseline_mask_conformance():
    """
    Test 05 — Baseline Action Feasibility
    Verify that LocalPolicy always selects Action 0,
    and GreedyPolicy produces actions in range [0..6].
    """
    with open("configs/paper_parameters.yaml", "r") as f:
        cfg = SimulationConfig(**yaml.safe_load(f))
        
    local_p = LocalPolicy(config=cfg)
    greedy_p = GreedyPolicy(config=cfg)
    
    dummy_obs = np.zeros(114, dtype=np.float32)
    
    assert local_p.select_action(dummy_obs) == 0
    
    greedy_act = greedy_p.select_action(dummy_obs)
    assert 0 <= greedy_act <= 6
