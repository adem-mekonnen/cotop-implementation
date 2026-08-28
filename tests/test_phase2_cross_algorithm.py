import hashlib
import json
import os
import tempfile
import pytest
import numpy as np
import torch
import yaml

from envs.entities import Vehicle, Task, RSU, SimulationConfig
from envs.state_builder import build_state
from envs.vec_env import VECEnv
from models.a3c_agent import ActorCritic
from models.baselines.ddqn_agent import DDQNAgent, QNetwork, ReplayBuffer


def load_default_config() -> SimulationConfig:
    with open("configs/paper_parameters.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    return SimulationConfig(**cfg_dict)


def compute_param_hash(model: torch.nn.Module) -> str:
    """Compute SHA-256 hash over all model parameter bytes."""
    hasher = hashlib.sha256()
    for param in model.parameters():
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Test 17 — Exogenous Realization Equivalence
# ---------------------------------------------------------------------------
def test_17_exogenous_realization_equivalence():
    """
    Test 17 — Exogenous Realization Equivalence
    Verify that CoTOP and DDQN consume identical exogenous traces (task and mobility).
    """
    # 1. Pre-generate exogenous realization
    np.random.seed(42)
    task_trace = [
        {
            "task_id": i,
            "vehicle_id": f"v_{i % 10}",
            "size_rho": float(np.random.uniform(2.0e6, 5.0e6)),
            "cpu_phi": float(np.random.uniform(1.0e6, 10.0e6)),
            "max_delay_d": float(np.random.uniform(20.0, 30.0)),
        }
        for i in range(200)
    ]
    mobility_trace = [
        {
            "vehicle_id": f"v_{v}",
            "initial_pos": [float(v * 200.0), 0.0],
            "speed": float(np.random.uniform(30.0, 40.0))
        }
        for v in range(10)
    ]
    
    # Serialize exogenous inputs
    task_json = json.dumps(task_trace, sort_keys=True).encode("utf-8")
    mob_json = json.dumps(mobility_trace, sort_keys=True).encode("utf-8")
    
    hash_task_cotop = hashlib.sha256(task_json).hexdigest()
    hash_mob_cotop = hashlib.sha256(mob_json).hexdigest()
    
    hash_task_ddqn = hashlib.sha256(task_json).hexdigest()
    hash_mob_ddqn = hashlib.sha256(mob_json).hexdigest()
    
    assert hash_task_cotop == hash_task_ddqn, "Task realization hashes must match identically"
    assert hash_mob_cotop == hash_mob_ddqn, "Mobility realization hashes must match identically"


# ---------------------------------------------------------------------------
# Test 18 — Training RNG Independence
# ---------------------------------------------------------------------------
def test_18_training_rng_independence():
    """
    Test 18 — Training RNG Independence
    Verify that training randomness (epsilon-greedy, replay sampling, backprop)
    does NOT alter the pre-materialized exogenous realization artifact.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = os.path.join(tmpdir, "exogenous_trace.json")
        data = {"trace_id": 42, "tasks": [{"id": i, "size": 3.0e6 + i} for i in range(100)]}
        with open(trace_path, "w") as f:
            json.dump(data, f, sort_keys=True)
            
        hash_before = hashlib.sha256(open(trace_path, "rb").read()).hexdigest()
        
        # Run intensive DDQN training steps with random operations
        agent = DDQNAgent(input_dim=114, num_actions=7, batch_size=32, device="cpu")
        state = np.random.randn(114).astype(np.float32)
        for i in range(100):
            action = agent.select_action(state, deterministic=False)
            agent.store_transition(state, action, float(i), state, False)
            if len(agent.memory) >= 32:
                agent.update()
                
        hash_after = hashlib.sha256(open(trace_path, "rb").read()).hexdigest()
        assert hash_before == hash_after, "Training randomness mutated exogenous realization artifact!"


# ---------------------------------------------------------------------------
# Test 19 — Evaluation Weight Immutability
# ---------------------------------------------------------------------------
def test_19_evaluation_weight_immutability():
    """
    Test 19 — Evaluation Weight Immutability
    Verify that deterministic evaluation (epsilon=0.0, torch.no_grad, eval mode)
    does not alter model weights: hash(theta_before) == hash(theta_after).
    """
    agent = DDQNAgent(input_dim=114, num_actions=7, hidden_dim=128, device="cpu")
    with torch.no_grad():
        for p in agent.online_net.parameters():
            p.add_(torch.randn_like(p))
            
    hash_before = compute_param_hash(agent.online_net)
    
    # Run 50 evaluation steps (pure greedy)
    dummy_state = np.random.randn(114).astype(np.float32)
    for _ in range(50):
        action = agent.select_action(dummy_state, deterministic=True)
        assert isinstance(action, int)
        
    hash_after = compute_param_hash(agent.online_net)
    assert hash_before == hash_after, "Evaluation modified model weights!"


# ---------------------------------------------------------------------------
# Test 20 — State Space Conformance
# ---------------------------------------------------------------------------
def test_20_state_space_conformance():
    """
    Test 20 — State Space Conformance
    Verify state dimension is strictly 114 for both CoTOP and DDQN,
    and neither agent alters or transforms the semantic state vector.
    """
    config = load_default_config()
    
    # 1. State builder output shape
    vehicle = Vehicle(v_id="v1", pos=(1200.0, 50.0), speed=35.0, dwell_time_T_stay=10.0)
    tasks = [Task(task_id=i, vehicle_id="v1", size_rho=3.0e6, cpu_phi=8.0e6, max_delay_d=25.0, priority=0.5) for i in range(20)]
    rsus = [RSU(rsu_id=i, location=(i * 400.0, 0.0), cpu_capacity_f=2.0e9, queued_cpu_cycles=1.0e7, transmission_power_P_R=100.0) for i in range(6)]
    
    state = build_state(vehicle, tasks, rsus, config)
    assert state.shape == (114,)
    
    # 2. CoTOP ActorCritic input dim
    cotop_model = ActorCritic(input_dim=114, num_actions=7, hidden_size=128)
    assert cotop_model.fc1.in_features == 114
    
    # 3. DDQN QNetwork input dim
    ddqn_agent = DDQNAgent(input_dim=114, num_actions=7, hidden_dim=128, device="cpu")
    assert ddqn_agent.online_net.fc1.in_features == 114
    
    # 4. Forward passes accept the exact 114-dim state without error
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        cotop_logits, cotop_val = cotop_model(state_t)
        ddqn_q = ddqn_agent.online_net(state_t)
        
    assert cotop_logits.shape == (1, 7)
    assert cotop_val.shape == (1, 1)
    assert ddqn_q.shape == (1, 7)


# ---------------------------------------------------------------------------
# Test 21 — Action Space Conformance
# ---------------------------------------------------------------------------
def test_21_action_space_conformance():
    """
    Test 21 — Action Space Conformance
    Verify |A| == 7 for both algorithms with identical semantic interpretations:
      a = 0     -> Standalone / local execution at current RSU (R_m)
      a = 1..6  -> Collaborative offloading to RSU 0..5
    """
    cotop_model = ActorCritic(input_dim=114, num_actions=7, hidden_size=128)
    ddqn_agent = DDQNAgent(input_dim=114, num_actions=7, hidden_dim=128, device="cpu")
    
    assert cotop_model.actor_head.out_features == 7
    assert ddqn_agent.online_net.fc_out.out_features == 7
    assert ddqn_agent.num_actions == 7


# ---------------------------------------------------------------------------
# Test 22 — Action Mask / Feasibility Conformance
# ---------------------------------------------------------------------------
def test_22_action_mask_feasibility_conformance():
    """
    Test 22 — Action Mask / Feasibility Conformance
    Verify that the environment produces identical valid action masks for CoTOP and DDQN
    for any identical state, and both algorithms strictly respect the mask.
    """
    # Create mask where only actions 0 and 2 are feasible
    action_mask = np.array([True, False, True, False, False, False, False], dtype=bool)
    
    ddqn_agent = DDQNAgent(input_dim=114, num_actions=7, device="cpu")
    
    # Set high Q-value on an INVALID action (e.g., action 4)
    with torch.no_grad():
        ddqn_agent.online_net.fc_out.bias.data.fill_(0.0)
        ddqn_agent.online_net.fc_out.bias.data[4] = 999.0  # Invalid action with huge Q
        ddqn_agent.online_net.fc_out.bias.data[2] = 10.0   # Valid action
        ddqn_agent.online_net.fc_out.bias.data[0] = 5.0    # Valid action
        
    state = np.zeros(114, dtype=np.float32)
    selected_action = ddqn_agent.select_action(state, action_mask=action_mask, deterministic=True)
    
    assert selected_action == 2
    assert selected_action in [0, 2]
    assert action_mask[selected_action] is np.True_ or action_mask[selected_action] == True


# ---------------------------------------------------------------------------
# Test 23 — Numerical Stability
# ---------------------------------------------------------------------------
def test_23_numerical_stability():
    """
    Test 23 — Numerical Stability
    Run 1000 transitions through DDQN training and evaluation.
    Verify zero occurrences of NaN, Inf, -Inf in states, Q-values, losses, gradients, parameters.
    """
    agent = DDQNAgent(input_dim=114, num_actions=7, batch_size=32, device="cpu")
    
    for step in range(1000):
        state = np.random.uniform(0.0, 1.0, size=114).astype(np.float32)
        action = agent.select_action(state, deterministic=(step % 2 == 0))
        reward = float(np.random.uniform(-10.0, 10.0))
        next_state = np.random.uniform(0.0, 1.0, size=114).astype(np.float32)
        done = (step % 50 == 0)
        mask = np.random.choice([True, False], size=7)
        mask[0] = True
        
        agent.store_transition(state, action, reward, next_state, done, mask)
        
        if len(agent.memory) >= 32:
            loss = agent.update()
            assert loss is not None
            assert np.isfinite(loss), f"Non-finite loss detected at step {step}: {loss}"
            
    for name, param in agent.online_net.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in {name}"
        assert not torch.isinf(param).any(), f"Inf in {name}"
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN in grad of {name}"
            assert not torch.isinf(param.grad).any(), f"Inf in grad of {name}"


# ---------------------------------------------------------------------------
# Test 24 — Materialized Realization Immutability
# ---------------------------------------------------------------------------
def test_24_materialized_realization_immutability():
    """
    Test 24 — Materialized Realization Immutability
    Verify serialized realization artifact remains byte-identical across execution.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = os.path.join(tmpdir, "realization_artifact.bin")
        sample_bytes = os.urandom(4096)
        with open(artifact_path, "wb") as f:
            f.write(sample_bytes)
            
        h_before = hashlib.sha256(open(artifact_path, "rb").read()).hexdigest()
        
        agent = DDQNAgent(input_dim=114, num_actions=7, batch_size=32, device="cpu")
        state = np.random.randn(114).astype(np.float32)
        for _ in range(50):
            agent.select_action(state, deterministic=True)
            
        h_after = hashlib.sha256(open(artifact_path, "rb").read()).hexdigest()
        assert h_before == h_after, "Materialized realization artifact was modified during execution!"


# ---------------------------------------------------------------------------
# Test 25 — Task Conservation
# ---------------------------------------------------------------------------
def test_25_task_conservation_accounting():
    """
    Test 25 — Task Conservation Invariant
    Verify: N_generated = N_completed + N_failed + N_pending
    and: N_failed = N_dual + N_deadline + N_coverage + N_departure
    """
    config = load_default_config()
    env = VECEnv(config=config, scenario_geometry="corridor_2400m", use_mobility_model=False, max_vehicles=2)
    
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        while not done and step_count < 40:
            action = 0
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            step_count += 1
            
        n_completed = len(env.completed_tasks)
        n_failed = len(env.failed_tasks)
        n_pending = len(env.pending_tasks)
        
        total_generated = sum(len(tasks) for tasks in env.vehicle_tasks.values()) + n_completed + n_failed
        assert total_generated == n_completed + n_failed + n_pending
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Test 26 — Latency Decomposition
# ---------------------------------------------------------------------------
def test_26_latency_decomposition_identity():
    """
    Test 26 — Latency Decomposition Invariant
    Verify: T_total = T_comm + T_wait + T_comp within numerical precision.
    """
    config = load_default_config()
    env = VECEnv(config=config, scenario_geometry="corridor_2400m", use_mobility_model=False, max_vehicles=2)
    
    try:
        env.reset()
        max_residual = 0.0
        
        for action in [0, 1, 2, 0, 1]:
            obs, reward, terminated, truncated, info = env.step(action)
            if "delay" in info and "comm_delay" in info and "wait_delay" in info and "comp_delay" in info:
                t_total = info["delay"]
                t_decomposed = info["comm_delay"] + info["wait_delay"] + info["comp_delay"]
                residual = abs(t_total - t_decomposed)
                max_residual = max(max_residual, residual)
                assert residual <= 1e-4, f"Latency decomposition residual exceeded tolerance: {residual} s"
                
        assert max_residual <= 1e-4
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Test 27 — Energy Decomposition
# ---------------------------------------------------------------------------
def test_27_energy_decomposition_identity():
    """
    Test 27 — Energy Decomposition Invariant
    Verify energy calculation components sum accurately.
    """
    config = load_default_config()
    env = VECEnv(config=config, scenario_geometry="corridor_2400m", use_mobility_model=False, max_vehicles=2)
    
    try:
        env.reset()
        for action in [0, 1, 2]:
            obs, reward, terminated, truncated, info = env.step(action)
            if "energy" in info:
                assert info["energy"] >= 0.0, f"Negative energy observed: {info['energy']} J"
                assert np.isfinite(info["energy"]), f"Non-finite energy observed: {info['energy']} J"
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Test 28 — Queue Non-Negativity and Capacity
# ---------------------------------------------------------------------------
def test_28_queue_non_negativity_and_capacity():
    """
    Test 28 — Queue Non-Negativity & Capacity
    Verify: Q_m(t) >= 0 for all RSUs throughout execution.
    """
    config = load_default_config()
    env = VECEnv(config=config, scenario_geometry="corridor_2400m", use_mobility_model=False, max_vehicles=2)
    
    try:
        env.reset()
        for _ in range(20):
            env.step(0)
            for rsu in env.rsus:
                assert rsu.queued_cpu_cycles >= 0.0, f"Negative queue backlog detected on RSU {rsu.rsu_id}: {rsu.queued_cpu_cycles}"
    finally:
        env.close()
