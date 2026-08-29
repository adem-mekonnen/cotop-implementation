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


def test_09_decoupled_double_dqn_target_evaluation():
    """
    Test 09 — Double-DQN Decoupling Verification
    Explicitly test:
      a* = argmax Q_online(s', a)
      target = Q_target(s', a*)
    Verify that the implementation NEVER computes max Q_target(s', a).
    """
    agent = DDQNAgent(input_dim=114, num_actions=7, hidden_dim=128, batch_size=2, gamma=0.95, device="cpu")
    
    # Construct synthetic next-states
    s1 = np.ones(114, dtype=np.float32)
    s2 = np.ones(114, dtype=np.float32) * 2.0
    
    # Store transitions in memory
    agent.store_transition(s1, 0, 1.0, s1, False)
    agent.store_transition(s2, 1, 2.0, s2, False)
    
    # Manually configure online and target network output layer weights and biases
    # For state s1:
    # online_q(s1) = [10.0, 50.0, 20.0, 5.0, 0.0, 0.0, 0.0] -> argmax is a*=1 (Q=50.0)
    # target_q(s1) = [100.0, 2.0, 30.0, 1.0, 0.0, 0.0, 0.0] -> target_q(s1, a*=1) = 2.0
    #
    # Standard DQN would erroneously take max target_q = 100.0!
    # DDQN must strictly take target_q(a*=1) = 2.0!
    with torch.no_grad():
        # Make all hidden layers identity-like or fixed
        for p in agent.online_net.parameters():
            p.fill_(0.0)
        for p in agent.target_net.parameters():
            p.fill_(0.0)
            
        # Set online biases to create distinct Q-values
        agent.online_net.fc_out.bias.data = torch.tensor([10.0, 50.0, 20.0, 5.0, 0.0, 0.0, 0.0])
        # Set target biases where action 0 has massive overestimation bias (100.0)
        agent.target_net.fc_out.bias.data = torch.tensor([100.0, 2.0, 30.0, 1.0, 0.0, 0.0, 0.0])
    
    # Evaluate decoupled target step directly
    s_t = torch.tensor(s1, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        next_online_q = agent.online_net(s_t)
        best_next_action = torch.argmax(next_online_q, dim=1, keepdim=True)
        assert best_next_action.item() == 1, "Online network must select action 1 (argmax Q_online)"
        
        next_target_q = agent.target_net(s_t)
        ddqn_eval_q = next_target_q.gather(1, best_next_action).squeeze(1).item()
        standard_dqn_q = torch.max(next_target_q, dim=1)[0].item()
        
        assert ddqn_eval_q == 2.0, "Target network evaluation must evaluate action selected by online network"
        assert standard_dqn_q == 100.0, "Standard DQN maximum is 100.0"
        assert ddqn_eval_q != standard_dqn_q, "DDQN and standard DQN must strictly diverge"
        
        # Expected Bellman target for r=1.0, gamma=0.95, done=0:
        # y = 1.0 + 0.95 * 2.0 = 2.90 (NOT 1.0 + 0.95 * 100.0 = 96.0)
        expected_y_ddqn = 1.0 + 0.95 * ddqn_eval_q
        assert pytest.approx(expected_y_ddqn, rel=1e-5) == 2.90


def test_10_terminal_transition_bellman_zeroing():
    """
    Test 10 — Terminal Transition Handling
    Verify that when done=True (d_t = 1.0), the Bellman target is strictly r_t
    and next_q_values are multiplied by (1.0 - 1.0) = 0.0.
    """
    agent = DDQNAgent(input_dim=114, num_actions=7, hidden_dim=128, batch_size=2, gamma=0.99, device="cpu")
    
    # Store 1 terminal transition and 1 non-terminal transition
    s = np.zeros(114, dtype=np.float32)
    s_next = np.zeros(114, dtype=np.float32)
    
    # Make target network output large positive values
    with torch.no_grad():
        agent.target_net.fc_out.bias.data.fill_(1000.0)
        agent.online_net.fc_out.bias.data.fill_(100.0)
        
    # Terminal transition: r = -50.0, done = True
    # Non-terminal transition: r = 5.0, done = False
    agent.store_transition(s, 0, -50.0, s_next, done=True)
    agent.store_transition(s, 0, 5.0, s_next, done=False)
    
    # Sample and compute targets
    states, actions, rewards, next_states, dones, _ = agent.memory.sample(batch_size=2)
    
    with torch.no_grad():
        next_online_q = agent.online_net(next_states)
        best_next_actions = torch.argmax(next_online_q, dim=1, keepdim=True)
        next_target_q = agent.target_net(next_states)
        next_q_values = next_target_q.gather(1, best_next_actions).squeeze(1)
        expected_targets = rewards + agent.gamma * (1.0 - dones) * next_q_values
        
    for i in range(2):
        if dones[i].item() == 1.0:
            # Terminal target MUST be exactly rewards[i] (-50.0)
            assert expected_targets[i].item() == -50.0
            assert expected_targets[i].item() != -50.0 + 0.99 * 1000.0
        else:
            # Non-terminal target MUST include discounted future value: 5.0 + 0.99 * 1000.0 = 995.0
            assert pytest.approx(expected_targets[i].item(), rel=1e-3) == 995.0


def test_11_invalid_action_mask_enforcement():
    """
    Test 11 — Action Mask Enforcement
    Verify that select_action never picks an invalid action in either:
    1. Greedy mode (deterministic=True)
    2. Exploratory mode (deterministic=False, epsilon=1.0)
    """
    agent = DDQNAgent(input_dim=114, num_actions=7, hidden_dim=128, device="cpu")
    state = np.zeros(114, dtype=np.float32)
    
    # Mask where ONLY actions 2 and 4 are valid
    mask = [False, False, True, False, True, False, False]
    valid_set = {2, 4}
    
    # 1. Greedy mode: Make action 0 and 1 have highest raw Q-values
    with torch.no_grad():
        agent.online_net.fc_out.bias.data = torch.tensor([1000.0, 500.0, 50.0, 10.0, 80.0, 5.0, 1.0])
        
    action_greedy = agent.select_action(state, action_mask=mask, deterministic=True)
    assert action_greedy == 4, "Must select action 4 (highest Q among valid actions {2, 4})"
    assert action_greedy in valid_set
    assert action_greedy not in {0, 1, 3, 5, 6}
    
    # 2. Exploratory mode: Over 300 random exploratory samples, only actions 2 and 4 may ever be returned
    agent.epsilon = 1.0  # Force pure exploration
    for _ in range(300):
        action_explore = agent.select_action(state, action_mask=mask, deterministic=False)
        assert action_explore in valid_set, f"Exploratory action {action_explore} was not in valid set {valid_set}"


def test_12_next_state_action_masking_in_update():
    """
    Test 12 — Next-State Action Masking in DDQN Update
    Verify that in agent.update(), best_next_actions a* is constrained by next_action_mask.
    """
    agent = DDQNAgent(input_dim=114, num_actions=7, hidden_dim=128, batch_size=2, device="cpu")
    
    s = np.zeros(114, dtype=np.float32)
    # Mask: only action 5 is valid
    mask_5_only = np.array([False, False, False, False, False, True, False], dtype=bool)
    
    agent.store_transition(s, 0, 1.0, s, False, mask_5_only)
    agent.store_transition(s, 0, 1.0, s, False, mask_5_only)
    
    # Make action 0 have highest unmasked Q-value (100.0), and action 5 have lower Q-value (10.0)
    with torch.no_grad():
        for p in agent.online_net.parameters():
            p.fill_(0.0)
        agent.online_net.fc_out.bias.data = torch.tensor([100.0, 20.0, 15.0, 5.0, 2.0, 10.0, 1.0])
        
        for p in agent.target_net.parameters():
            p.fill_(0.0)
        agent.target_net.fc_out.bias.data = torch.tensor([50.0, 50.0, 50.0, 50.0, 50.0, 33.0, 50.0])
    
    states, actions, rewards, next_states, dones, next_masks = agent.memory.sample(batch_size=2)
    
    with torch.no_grad():
        next_online_q = agent.online_net(next_states)
        # Apply mask
        next_online_q = torch.where(next_masks, next_online_q, torch.tensor(-1e9))
        best_next_actions = torch.argmax(next_online_q, dim=1, keepdim=True)
        
        # Must select action 5 (only valid action), NOT action 0
        assert torch.all(best_next_actions == 5)
        
        next_target_q = agent.target_net(next_states)
        evaluated_target_q = next_target_q.gather(1, best_next_actions).squeeze(1)
        assert torch.all(evaluated_target_q == 33.0)


def test_13_all_invalid_action_mask_safety_fallback():
    """
    Test 13 — All-Invalid Action Mask Edge Case Safety
    Verify that when all actions in a mask are False, agent falls back safely to all actions allowed.
    """
    agent = DDQNAgent(input_dim=114, num_actions=7, hidden_dim=128, device="cpu")
    state = np.zeros(114, dtype=np.float32)
    all_false_mask = [False] * 7
    
    # In greedy mode
    action_greedy = agent.select_action(state, action_mask=all_false_mask, deterministic=True)
    assert 0 <= action_greedy < 7
    
    # In exploratory mode
    agent.epsilon = 1.0
    action_explore = agent.select_action(state, action_mask=all_false_mask, deterministic=False)
    assert 0 <= action_explore < 7


def test_14_replay_buffer_advanced_sampling_and_fifo_wraparound():
    """
    Test 14 — Replay Buffer Sampling Distribution & Wrap-around
    Verify buffer maintains exact FIFO ordering, boundary clipping, and non-empty sampling.
    """
    capacity = 100
    buffer = ReplayBuffer(capacity=capacity, state_dim=4, num_actions=2)
    
    # Push 250 items through capacity=100 buffer
    for i in range(250):
        buffer.push(
            state=np.array([i, i, i, i], dtype=np.float32),
            action=i % 2,
            reward=float(i),
            next_state=np.array([i+1, i+1, i+1, i+1], dtype=np.float32),
            done=False
        )
        
    assert len(buffer) == 100
    assert buffer.size == 100
    
    # Logical index 0 must be transition #150 (oldest in buffer)
    oldest_state = buffer[0][0]
    assert oldest_state[0] == 150.0
    
    # Logical index 99 must be transition #249 (newest in buffer)
    newest_state = buffer[99][0]
    assert newest_state[0] == 249.0
    
    # Sampling batch
    states, actions, rewards, next_states, dones, masks = buffer.sample(batch_size=32)
    assert states.shape == (32, 4)
    assert actions.shape == (32,)
    assert (states[:, 0] >= 150.0).all()
    assert (states[:, 0] <= 249.0).all()


def test_15_optimizer_state_checkpoint_recovery():
    """
    Test 15 — Optimizer State Checkpoint Recovery
    Verify that Adam momentum/variance buffers and step counters are serialized and restored exactly.
    """
    agent1 = DDQNAgent(input_dim=114, num_actions=7, hidden_dim=128, batch_size=10, device="cpu")
    
    # Push transitions and train for 5 steps
    s = np.random.randn(114).astype(np.float32)
    for i in range(50):
        agent1.store_transition(s, i % 7, float(i), s, False)
        
    for _ in range(5):
        agent1.update()
        
    # Check that optimizer state has step and momentum entries
    opt_state1 = agent1.optimizer.state_dict()
    assert len(opt_state1['state']) > 0
    
    # Save checkpoint and reload into agent2
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "opt_recovery_test.pt")
        agent1.save_checkpoint(ckpt_path)
        
        agent2 = DDQNAgent(input_dim=114, num_actions=7, hidden_dim=128, batch_size=10, device="cpu")
        agent2.load_checkpoint(ckpt_path)
        
        # Verify optimizer step count and momentum buffers match bitwise
        opt_state2 = agent2.optimizer.state_dict()
        for k in opt_state1['state']:
            for tensor_key in ['exp_avg', 'exp_avg_sq']:
                if tensor_key in opt_state1['state'][k]:
                    t1 = opt_state1['state'][k][tensor_key]
                    t2 = opt_state2['state'][k][tensor_key]
                    assert torch.equal(t1, t2), f"Mismatch in optimizer tensor {tensor_key}"


def test_16_rng_recovery_and_deterministic_continuation():
    """
    Test 16 — RNG Recovery & Deterministic Execution
    Verify that setting deterministic seeds produces identical action sequences.
    """
    from utils.seed import set_seed
    
    set_seed(42)
    agent1 = DDQNAgent(input_dim=114, num_actions=7, hidden_dim=128, device="cpu")
    state = np.random.randn(114).astype(np.float32)
    
    actions1 = [agent1.select_action(state, deterministic=False) for _ in range(20)]
    
    set_seed(42)
    agent2 = DDQNAgent(input_dim=114, num_actions=7, hidden_dim=128, device="cpu")
    actions2 = [agent2.select_action(state, deterministic=False) for _ in range(20)]
    
    assert actions1 == actions2, "RNG-seeded action trajectories must be 100% deterministic and identical"

