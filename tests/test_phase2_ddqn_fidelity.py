import os
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn

from models.baselines.ddqn_agent import DDQNAgent, QNetwork, ReplayBuffer


def test_01_ddqn_action_selection():
    """
    Test 01 — DDQN Action Selection
    Verify greedy action selection performs argmax(Q_online) and respects action mask.
    """
    agent = DDQNAgent(input_dim=114, num_actions=7, hidden_dim=128, device="cpu")
    
    # Create deterministic input state
    state = np.zeros(114, dtype=np.float32)
    state[0] = 0.5
    
    # 1. Without mask: greedy action matches argmax of online network
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        raw_q = agent.online_net(state_t).squeeze(0).numpy()
        expected_action = int(np.argmax(raw_q))
        
    chosen_action = agent.select_action(state, action_mask=None, deterministic=True)
    assert chosen_action == expected_action
    
    # 2. With mask: manually construct scenario where highest Q-value action is masked out
    with torch.no_grad():
        # Set weights such that action 3 has highest Q, action 1 has second highest
        agent.online_net.fc_out.bias.data.fill_(0.0)
        agent.online_net.fc_out.bias.data[3] = 100.0  # Highest
        agent.online_net.fc_out.bias.data[1] = 50.0   # Second highest
        agent.online_net.fc_out.bias.data[0] = 10.0
    
    # Mask action 3 out (action 3 is invalid)
    mask = [True, True, False, False, False, False, False]  # Only actions 0 and 1 valid
    masked_action = agent.select_action(state, action_mask=mask, deterministic=True)
    
    # Must choose action 1 (highest valid action), NOT action 3
    assert masked_action == 1
    assert masked_action != 3


def test_02_ddqn_mathematical_toy_test():
    """
    Test 02 — DDQN Mathematical Toy Test
    Analytical verification of decoupled Double Q-learning target vs standard DQN.
    
    Given:
      Q_online(s', :) = [1.0, 5.0, 3.0] -> argmax = index 1 (action 2)
      Q_target(s', :) = [9.0, 2.0, 7.0]
      Reward r = 1.5, gamma = 0.99, done = 0
      
    DDQN target: 1.5 + 0.99 * 2.0 = 3.48
    Standard DQN target: 1.5 + 0.99 * 9.0 = 10.41
    """
    agent = DDQNAgent(input_dim=3, num_actions=3, hidden_dim=16, gamma=0.99, device="cpu")
    
    q_online = torch.tensor([[1.0, 5.0, 3.0]], dtype=torch.float32)
    q_target = torch.tensor([[9.0, 2.0, 7.0]], dtype=torch.float32)
    reward = torch.tensor([1.5], dtype=torch.float32)
    done = torch.tensor([0.0], dtype=torch.float32)
    gamma = 0.99
    
    # 1. Online network selects best action a*
    best_action = torch.argmax(q_online, dim=1, keepdim=True)
    assert best_action.item() == 1  # 0-based index for second action
    
    # 2. Target network evaluates that chosen action
    evaluated_target_q = q_target.gather(1, best_action).squeeze(1)
    assert evaluated_target_q.item() == 2.0
    
    # 3. Construct target
    y_ddqn = reward + gamma * (1.0 - done) * evaluated_target_q
    assert pytest.approx(y_ddqn.item(), rel=1e-5) == 3.48
    
    # 4. Standard DQN would erroneously take max(q_target) = 9.0
    y_dqn = reward + gamma * (1.0 - done) * torch.max(q_target, dim=1)[0]
    assert pytest.approx(y_dqn.item(), rel=1e-5) == 10.41
    assert y_ddqn.item() != y_dqn.item()


def test_03_target_network_synchronization():
    """
    Test 03 — Target Network Synchronization
    Verify target == online at init, modification creates diff, and sync restores equality.
    Verify target_update_frequency is exactly 100 steps.
    """
    agent = DDQNAgent(input_dim=114, num_actions=7, hidden_dim=128, target_update_frequency=100, device="cpu")
    assert agent.target_update_frequency == 100
    
    # 1. Initial parameters must be identical
    for p_on, p_tgt in zip(agent.online_net.parameters(), agent.target_net.parameters()):
        assert torch.equal(p_on, p_tgt)
        
    # 2. Deliberately modify online network parameters
    with torch.no_grad():
        for p in agent.online_net.parameters():
            p.add_(torch.randn_like(p) + 1.0)
            
    # Target network must now differ
    has_diff = False
    for p_on, p_tgt in zip(agent.online_net.parameters(), agent.target_net.parameters()):
        if not torch.equal(p_on, p_tgt):
            has_diff = True
            break
    assert has_diff, "Target network should differ from online network before sync"
    
    # 3. Trigger target synchronization
    agent.sync_target_network()
    
    # All parameters must now be strictly equal
    for p_on, p_tgt in zip(agent.online_net.parameters(), agent.target_net.parameters()):
        assert torch.equal(p_on, p_tgt), "Target network must match online network after sync"


def test_04_replay_buffer_fifo_and_shapes():
    """
    Test 04 — Replay Buffer
    Verify capacity = 10,000, FIFO eviction, sampling shapes, and minimum capacity check.
    """
    capacity = 10000
    buffer = ReplayBuffer(capacity=capacity)
    assert buffer.capacity == 10000
    assert len(buffer) == 0
    
    # 1. Push transitions up to capacity + 50 to test FIFO eviction
    state_proto = np.ones(114, dtype=np.float32)
    next_state_proto = np.ones(114, dtype=np.float32) * 2.0
    mask_proto = np.array([True] * 7, dtype=bool)
    
    for i in range(capacity + 50):
        buffer.push(
            state=state_proto * i,
            action=i % 7,
            reward=float(i),
            next_state=next_state_proto * i,
            done=(i % 10 == 0),
            next_action_mask=mask_proto
        )
        
    assert len(buffer) == capacity
    
    # Oldest 50 transitions (0..49) must have been evicted
    # The first item in the buffer must be transition #50
    first_state = buffer.buffer[0][0]
    assert np.allclose(first_state, state_proto * 50)
    
    # 2. Minibatch sampling shapes
    batch_size = 64
    states, actions, rewards, next_states, dones, next_masks = buffer.sample(batch_size=batch_size)
    
    assert states.shape == (64, 114)
    assert actions.shape == (64,)
    assert rewards.shape == (64,)
    assert next_states.shape == (64, 114)
    assert dones.shape == (64,)
    assert next_masks.shape == (64, 7)
    
    # 3. Agent update returns None when buffer < batch_size
    fresh_agent = DDQNAgent(input_dim=114, num_actions=7, batch_size=64, device="cpu")
    for _ in range(30):
        fresh_agent.store_transition(state_proto, 0, 1.0, next_state_proto, False)
    assert len(fresh_agent.memory) == 30
    assert fresh_agent.update() is None


def test_05_loss_computation_smooth_l1():
    """
    Test 05 — Loss Computation
    Verify Smooth L1 (Huber) loss calculation against analytical hand-calculated tensor ground truth (0.8125).
    """
    agent = DDQNAgent(input_dim=114, num_actions=7, device="cpu")
    
    # Target: [3.0, 5.0], Predicted: [2.5, 3.0] -> Errors: [0.5, 2.0]
    # Smooth L1 for beta=1.0:
    # 0.5 * (0.5)^2 = 0.125
    # 2.0 - 0.5 = 1.500
    # Mean = (0.125 + 1.500) / 2 = 0.8125
    y_target = torch.tensor([3.0, 5.0], dtype=torch.float32)
    q_pred = torch.tensor([2.5, 3.0], dtype=torch.float32)
    
    computed_loss = agent.loss_fn(q_pred, y_target)
    assert pytest.approx(computed_loss.item(), rel=1e-5) == 0.8125


def test_06_gradient_flow_isolation():
    """
    Test 06 — Gradient Flow Isolation
    Verify online parameters update and receive gradients; target parameters receive no gradients.
    """
    agent = DDQNAgent(input_dim=114, num_actions=7, batch_size=64, device="cpu")
    
    # Fill replay buffer with sufficient transitions
    state = np.random.randn(114).astype(np.float32)
    next_state = np.random.randn(114).astype(np.float32)
    mask = np.array([True] * 7, dtype=bool)
    for _ in range(100):
        agent.store_transition(state, np.random.randint(0, 7), 1.0, next_state, False, mask)
        
    # Record initial weights
    initial_online_weights = [p.clone().detach() for p in agent.online_net.parameters()]
    initial_target_weights = [p.clone().detach() for p in agent.target_net.parameters()]
    
    # Perform update step
    loss = agent.update()
    assert loss is not None
    assert np.isfinite(loss)
    
    # Verify online parameters changed
    online_changed = False
    for p_init, p_curr in zip(initial_online_weights, agent.online_net.parameters()):
        if not torch.equal(p_init, p_curr):
            online_changed = True
            break
    assert online_changed, "Online network weights must update after optimizer step"
    
    # Verify target parameters remained completely unchanged
    for p_init, p_curr in zip(initial_target_weights, agent.target_net.parameters()):
        assert torch.equal(p_init, p_curr), "Target network weights must remain frozen during update"
        assert p_curr.grad is None or not p_curr.requires_grad


def test_07_epsilon_schedule_bounds_and_decay():
    """
    Test 07 — Epsilon Schedule
    Verify epsilon_start=1.0, epsilon_end=0.05, decay horizon=200 episodes.
    Verify monotonic non-increasing decay and boundary adherence.
    """
    agent = DDQNAgent(
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=200,
        device="cpu"
    )
    
    epsilons = []
    for ep in range(300):
        eps = agent.compute_epsilon(ep)
        epsilons.append(eps)
        
    assert pytest.approx(epsilons[0], rel=1e-5) == 1.0
    assert pytest.approx(epsilons[100], rel=1e-5) == 0.525
    assert pytest.approx(epsilons[200], rel=1e-5) == 0.05
    assert pytest.approx(epsilons[250], rel=1e-5) == 0.05
    assert pytest.approx(epsilons[299], rel=1e-5) == 0.05
    
    # Monotonic non-increasing
    for i in range(len(epsilons) - 1):
        assert epsilons[i] >= epsilons[i + 1]
        assert epsilons[i] >= 0.05
        assert epsilons[i] <= 1.0


def test_08_checkpoint_exact_recovery():
    """
    Test 08 — Checkpoint Exact Recovery
    Save DDQN checkpoint, load into fresh instance, verify exact equality of all components.
    """
    agent1 = DDQNAgent(input_dim=114, num_actions=7, hidden_dim=128, device="cpu")
    agent1.train_step_count = 142
    agent1.set_episode(75)
    
    # Modify online network to non-trivial weights
    with torch.no_grad():
        for p in agent1.online_net.parameters():
            p.add_(torch.randn_like(p))
    agent1.sync_target_network()
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "ddqn_test_ckpt.pt")
        agent1.save_checkpoint(ckpt_path, extra_metadata={"test_run": "unit_test_08"})
        
        # Fresh agent
        agent2 = DDQNAgent(input_dim=114, num_actions=7, hidden_dim=128, device="cpu")
        agent2.load_checkpoint(ckpt_path)
        
        # Verify step and episode counters
        assert agent2.train_step_count == 142
        assert agent2.episode_count == 75
        assert pytest.approx(agent2.epsilon, rel=1e-5) == agent1.epsilon
        
        # Verify online network parameters match bitwise
        for p1, p2 in zip(agent1.online_net.parameters(), agent2.online_net.parameters()):
            assert torch.equal(p1, p2)
            
        # Verify target network parameters match bitwise
        for p1, p2 in zip(agent1.target_net.parameters(), agent2.target_net.parameters()):
            assert torch.equal(p1, p2)
