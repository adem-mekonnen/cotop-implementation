import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def test_ddqn_decoupled_target_toy_calculation():
    """
    Mathematical Toy Test 1: DDQN Decoupled Target Selection
    Verifies that target calculation evaluates Q_target at argmax(Q_online),
    NOT max(Q_target).
    
    Given:
      Q_online(s', :) = [1.0, 5.0, 3.0] -> argmax = index 1
      Q_target(s', :) = [9.0, 2.0, 7.0]
      Reward r = 1.5, gamma = 0.9, terminal d = 0
      
    DDQN Target:
      y = 1.5 + 0.9 * (1 - 0) * Q_target(s', 1) = 1.5 + 0.9 * 2.0 = 3.3
      
    DQN (Standard / Overestimated) Target:
      y_dqn = 1.5 + 0.9 * 9.0 = 9.6
    """
    q_online = torch.tensor([[1.0, 5.0, 3.0]], dtype=torch.float32)
    q_target = torch.tensor([[9.0, 2.0, 7.0]], dtype=torch.float32)
    reward = torch.tensor([1.5], dtype=torch.float32)
    gamma = 0.9
    done = torch.tensor([0.0], dtype=torch.float32)
    
    # 1. Action selection using online network
    best_actions = torch.argmax(q_online, dim=1, keepdim=True)  # tensor([[1]])
    assert best_actions.item() == 1
    
    # 2. Target evaluation using target network at best_actions
    selected_target_q = q_target.gather(1, best_actions).squeeze(1)  # tensor([2.0])
    assert selected_target_q.item() == 2.0
    
    # 3. Bellman target calculation
    ddqn_target = reward + gamma * (1.0 - done) * selected_target_q
    assert pytest.approx(ddqn_target.item(), rel=1e-5) == 3.3
    
    # 4. Prove strictly decoupled from standard max(q_target) = 9.0
    dqn_target = reward + gamma * (1.0 - done) * torch.max(q_target, dim=1)[0]
    assert pytest.approx(dqn_target.item(), rel=1e-5) == 9.6
    assert ddqn_target.item() != dqn_target.item()

def test_ddqn_smooth_l1_huber_toy_calculation():
    """
    Mathematical Toy Test 2: Smooth L1 (Huber) Loss Analytical Verification
    
    Huber loss formula (beta=1.0):
      l(u) = 0.5 * u^2       if |u| < 1.0
      l(u) = |u| - 0.5       if |u| >= 1.0
      
    For:
      Target y = [3.0, 5.0]
      Predicted Q = [2.5, 3.0]
      Error u = [0.5, 2.0]
      
      l(0.5) = 0.5 * 0.25 = 0.125
      l(2.0) = 2.0 - 0.5 = 1.500
      Mean loss = (0.125 + 1.500) / 2 = 0.8125
    """
    y_target = torch.tensor([3.0, 5.0], dtype=torch.float32)
    q_pred = torch.tensor([2.5, 3.0], dtype=torch.float32)
    
    loss_fn = nn.SmoothL1Loss(reduction='mean', beta=1.0)
    loss = loss_fn(q_pred, y_target)
    
    assert pytest.approx(loss.item(), rel=1e-5) == 0.8125

def test_quantile_expectation_toy_calculation():
    """
    Mathematical Toy Test 3: Quantile Regression Expectation
    Q(s, a) = (1 / K) * sum_{k=1}^K theta_k(s, a)
    
    For K=4 quantiles: [1.0, 2.0, 3.0, 4.0]
    Mean Q = 2.5
    """
    quantiles = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float32)  # shape (1, 1, 4)
    expected_q = torch.mean(quantiles, dim=-1)  # shape (1, 1)
    assert pytest.approx(expected_q.item(), rel=1e-5) == 2.5
