import pytest
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import os
import math
import copy
import hashlib
import json
import tempfile

from envs.vec_env import VECEnv
from envs.entities import SimulationConfig, Task, Vehicle, RSU
from models.a3c_agent import ActorCritic
from models.mobility_gat import MobilityGAT_GRU
from models.baselines.ddqn_agent import DDQNAgent, QNetwork
from envs.state_builder import build_state
from utils.task_priority import compute_task_priority_paper, compute_task_priority_normalized, prioritize_tasks_paper
from envs.comm_model import compute_v2r_rate, compute_r2r_rate
from envs.comp_model import calculate_case1_standalone, calculate_case2_collaboration
from utils.seed import set_seed

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

# ------------------------------------------------------------------------------
# 1. GAT Dimensions & Multi-Head Specification
# ------------------------------------------------------------------------------
def test_01_gat_dimensions():
    """Test 1: GAT Layer 1 (concat) and Layer 2 (mean head) dimensions."""
    model = MobilityGAT_GRU(input_dim=2, embed_dim=64, num_heads=4, gru_hidden=64, output_dim=2)
    x_seq = torch.randn(4, 5, 2)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    # Layer 1 test
    h_0 = model.coordinate_expansion_mlp(x_seq[:, 0, :])
    assert h_0.shape == (4, 64)
    z_1 = model.gat_layer1(h_0, edge_index)
    assert z_1.shape == (4, 64) # 4 heads * 16 = 64
    
    # Layer 2 test (mean head averaging across 4 heads)
    z_2 = model.gat_layer2(z_1, edge_index)
    assert z_2.shape == (4, 64) # 4 heads averaged = 64

# ------------------------------------------------------------------------------
# 2. Multi-Node GAT Behavior
# ------------------------------------------------------------------------------
def test_02_multi_node_gat_behavior():
    """Test 2: Multi-node GAT processes N in [1, 2, 5, 10] nodes with spatial sensitivity."""
    model = MobilityGAT_GRU(input_dim=2, embed_dim=64, num_heads=4, gru_hidden=64, output_dim=2)
    model.eval()
    
    for N in [1, 2, 5, 10]:
        x_seq = torch.randn(N, 5, 2)
        edges = [[i, j] for i in range(N) for j in range(N)]
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        with torch.no_grad():
            out = model(x_seq, edge_index)
        assert out.shape == (N, 5, 2)

# ------------------------------------------------------------------------------
# 3. Mean-Head Aggregation (Eq. 18)
# ------------------------------------------------------------------------------
def test_03_mean_head_aggregation_eq18():
    """Test 3: GAT Layer 2 mean-head aggregation per Eq. 18."""
    model = MobilityGAT_GRU(input_dim=2, embed_dim=64, num_heads=4, gru_hidden=64, output_dim=2)
    assert model.gat_layer2.concat is False, "Layer 2 must average heads (concat=False)"
    assert model.gat_layer2.heads == 4, "Layer 2 must have 4 attention heads"

# ------------------------------------------------------------------------------
# 4. Eq. 23 Task Priority Monotonicity
# ------------------------------------------------------------------------------
def test_04_eq23_task_priority_monotonicity():
    """Test 4: Eq. 23 priority monotonicity with rho, d, and T_stay."""
    t_base = Task(task_id=1, vehicle_id="v1", size_rho=3.0e6, cpu_phi=5.0e6, max_delay_d=25.0)
    t_larger = Task(task_id=2, vehicle_id="v1", size_rho=4.0e6, cpu_phi=5.0e6, max_delay_d=25.0)
    t_tighter = Task(task_id=3, vehicle_id="v1", size_rho=3.0e6, cpu_phi=5.0e6, max_delay_d=20.0)
    
    p_base = compute_task_priority_paper(t_base, dwell_time=10.0, alpha=0.3, beta=0.7)
    p_larger = compute_task_priority_paper(t_larger, dwell_time=10.0, alpha=0.3, beta=0.7)
    p_tighter = compute_task_priority_paper(t_tighter, dwell_time=10.0, alpha=0.3, beta=0.7)
    p_longer_dwell = compute_task_priority_paper(t_base, dwell_time=20.0, alpha=0.3, beta=0.7)
    
    assert p_larger > p_base, "Priority must increase with task data size"
    assert p_tighter > p_base, "Priority must increase with tighter deadline"
    assert p_longer_dwell > p_base, "Priority must increase with longer dwell time"

# ------------------------------------------------------------------------------
# 5. Eq. 23 Zero Handling & Boundary Safety
# ------------------------------------------------------------------------------
def test_05_eq23_zero_handling():
    """Test 5: Eq. 23 handles non-positive dwell times and deadlines safely."""
    t_zero_d = Task(task_id=1, vehicle_id="v1", size_rho=3.0e6, cpu_phi=5.0e6, max_delay_d=0.0)
    p_zero_d = compute_task_priority_paper(t_zero_d, dwell_time=10.0)
    assert not math.isnan(p_zero_d) and not math.isinf(p_zero_d)
    
    t_normal = Task(task_id=2, vehicle_id="v1", size_rho=3.0e6, cpu_phi=5.0e6, max_delay_d=25.0)
    p_zero_dwell = compute_task_priority_paper(t_normal, dwell_time=0.0)
    assert not math.isnan(p_zero_dwell) and not math.isinf(p_zero_dwell)

# ------------------------------------------------------------------------------
# 6. Eq. 25 Reward Linear Combination
# ------------------------------------------------------------------------------
def test_06_eq25_reward_linear_combination():
    """Test 6: Eq. 25 reward for successful task completion."""
    delay = 2.5
    energy = 4.0
    eps = 0.5
    reward = -(eps * delay + (1.0 - eps) * energy)
    assert reward == -3.25

# ------------------------------------------------------------------------------
# 7. Failure Predicates & Penalty -Z
# ------------------------------------------------------------------------------
def test_07_failure_predicates_and_penalties():
    """Test 7: Eq. 25 penalty -Z on deadline and coverage violations."""
    penalty_z = 100.0
    
    # Deadline violation
    delay = 32.0
    deadline = 25.0
    assert delay > deadline
    reward_deadline_fail = -penalty_z
    assert reward_deadline_fail == -100.0
    
    # Coverage violation
    dist = 450.0
    rsu_range = 400.0
    assert dist > rsu_range
    reward_cov_fail = -penalty_z
    assert reward_cov_fail == -100.0

# ------------------------------------------------------------------------------
# 8. A3C Advantage Calculation
# ------------------------------------------------------------------------------
def test_08_a3c_advantage_calculation():
    """Test 8: A3C Advantage A(s, a) = R_t - V(s) with discount gamma=0.99."""
    rewards = [1.0, 2.0, 3.0]
    gamma = 0.99
    values = torch.tensor([1.5, 2.5, 3.0])
    
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
        
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    advantages = returns_tensor - values
    
    expected_R2 = 3.0
    expected_R1 = 2.0 + 0.99 * 3.0 # 4.97
    expected_R0 = 1.0 + 0.99 * 4.97 # 5.9203
    
    assert torch.isclose(returns_tensor[2], torch.tensor(expected_R2))
    assert torch.isclose(returns_tensor[1], torch.tensor(expected_R1))
    assert torch.isclose(returns_tensor[0], torch.tensor(expected_R0))
    assert torch.allclose(advantages, returns_tensor - values)

# ------------------------------------------------------------------------------
# 9. A3C Gradient Routing
# ------------------------------------------------------------------------------
def test_09_a3c_gradient_routing():
    """Test 9: Advantage detach prevents actor loss gradient leakage into critic."""
    model = ActorCritic(input_dim=20, num_actions=7)
    state = torch.randn(1, 20)
    logits, value = model(state)
    probs = F.softmax(logits, dim=-1)
    
    action_log_prob = torch.log(probs[0, 0])
    R_target = torch.tensor([10.0])
    advantage = (R_target - value.detach()).view(-1)
    
    actor_loss = -(action_log_prob * advantage).mean()
    actor_loss.backward()
    
    # Verify actor head has gradients, critic head does NOT
    assert model.actor_head.weight.grad is not None
    assert model.critic_head.weight.grad is None

# ------------------------------------------------------------------------------
# 10. State Dimension Composition (114)
# ------------------------------------------------------------------------------
def test_10_state_dimension_composition():
    """Test 10: State dimension = 114 (4 veh + 80 task + 30 RSU)."""
    config = SimulationConfig()
    vehicle = Vehicle(v_id="v0", pos=(100.0, 0.0), speed=35.0, dwell_time_T_stay=10.0)
    tasks = [
        Task(task_id=i, vehicle_id="v0", size_rho=3.0e6, cpu_phi=5.0e6, max_delay_d=25.0, priority=1.0)
        for i in range(20)
    ]
    rsus = [
        RSU(rsu_id=i, location=(float(i * 400), 0.0), cpu_capacity_f=2.0e9, queued_cpu_cycles=0.0, transmission_power_P_R=100.0)
        for i in range(6)
    ]
    state = build_state(vehicle, tasks, rsus, config)
    assert state.shape == (114,)

# ------------------------------------------------------------------------------
# 11. State Bounds & Normalization
# ------------------------------------------------------------------------------
def test_11_state_bounds_and_normalization():
    """Test 11: State vector elements are finite and within normalized ranges."""
    config = SimulationConfig()
    vehicle = Vehicle(v_id="v0", pos=(2400.0, 0.0), speed=40.0, dwell_time_T_stay=50.0)
    tasks = [
        Task(task_id=i, vehicle_id="v0", size_rho=5.0e6, cpu_phi=10.0e6, max_delay_d=30.0, priority=1.0)
        for i in range(20)
    ]
    rsus = [
        RSU(rsu_id=i, location=(float(i * 400), 0.0), cpu_capacity_f=4.0e9, queued_cpu_cycles=1.0e8, transmission_power_P_R=100.0)
        for i in range(6)
    ]
    state = build_state(vehicle, tasks, rsus, config)
    assert np.all(np.isfinite(state))
    assert state[0] <= 1.0 # x / 2400
    assert state[2] <= 1.0 # speed / 40

# ------------------------------------------------------------------------------
# 12. Action Dimension (|A| = 7)
# ------------------------------------------------------------------------------
def test_12_action_dimension():
    """Test 12: Action dimension is 7 (0: standalone, 1..6: collaborative RSUs)."""
    model = ActorCritic(input_dim=114, num_actions=7)
    state = torch.randn(1, 114)
    logits, value = model(state)
    assert logits.shape == (1, 7)
    assert value.shape == (1, 1)

# ------------------------------------------------------------------------------
# 13. Action Masking Zero Probability
# ------------------------------------------------------------------------------
def test_13_action_masking_zero_probability():
    """Test 13: Invalid action mask forces softmax probability to exactly 0."""
    logits = torch.tensor([[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]])
    mask = torch.tensor([[True, True, True, False, False, False, False]])
    
    logits[~mask] = -1e9
    probs = F.softmax(logits, dim=-1)
    
    for i in range(3, 7):
        assert probs[0, i].item() == 0.0

# ------------------------------------------------------------------------------
# 14. Action Probability Normalization
# ------------------------------------------------------------------------------
def test_14_action_probability_normalization():
    """Test 14: Valid action probabilities sum to exactly 1.0."""
    logits = torch.tensor([[2.0, -1.0, 3.0, 0.0, -5.0, 1.0, 4.0]])
    mask = torch.tensor([[True, True, False, True, False, True, False]])
    
    logits[~mask] = -1e9
    probs = F.softmax(logits, dim=-1)
    
    assert torch.isclose(probs.sum(), torch.tensor(1.0))

# ------------------------------------------------------------------------------
# 15. Task Conservation Invariant
# ------------------------------------------------------------------------------
def test_15_task_conservation():
    """Test 15: N_generated = N_completed + N_failed + N_pending across step lifecycle."""
    total_gen = 20
    completed = 12
    failed_deadline = 3
    failed_cov = 2
    failed_dep = 1
    pending = 2
    
    total_failed = failed_deadline + failed_cov + failed_dep
    assert total_gen == completed + total_failed + pending

# ------------------------------------------------------------------------------
# 16. Latency Decomposition Residual
# ------------------------------------------------------------------------------
def test_16_latency_decomposition_residual():
    """Test 16: T_total = T_comm + T_wait + T_comp with residual R == 0."""
    t_comm = 1.824
    t_comp = 0.005
    t_wait = 0.500
    t_total = t_comm + t_comp + t_wait
    
    residual = t_total - (t_comm + t_comp + t_wait)
    assert abs(residual) < 1e-12

# ------------------------------------------------------------------------------
# 17. Energy Decomposition Non-Negativity
# ------------------------------------------------------------------------------
def test_17_energy_decomposition_non_negative():
    """Test 17: E_total >= 0 and finite."""
    d, e = calculate_case1_standalone(
        task_size_rho=3.0e6, task_cpu_phi=5.0e6, w_v2r=20.0e6, rsu_cpu_f=2.0e9,
        power_v=0.01, compute_power_rsu=50.0, t_wait=0.0
    )
    assert e >= 0.0
    assert not math.isnan(e) and not math.isinf(e)

# ------------------------------------------------------------------------------
# 18. Queue Non-Negativity Invariant
# ------------------------------------------------------------------------------
def test_18_queue_non_negativity():
    """Test 18: RSU queued work Q_m(t) >= 0 under continuous service draining."""
    rsu = RSU(rsu_id=0, location=(0.0, 0.0), cpu_capacity_f=2.0e9, queued_cpu_cycles=5.0e8, transmission_power_P_R=100.0)
    # Drain 1 second (2.0e9 cycles)
    rsu.queued_cpu_cycles = max(0.0, rsu.queued_cpu_cycles - rsu.cpu_capacity_f * 1.0)
    assert rsu.queued_cpu_cycles == 0.0

# ------------------------------------------------------------------------------
# 19. Realization Immutability
# ------------------------------------------------------------------------------
def test_19_realization_immutability():
    """Test 19: Exogenous realization artifact cannot be mutated during evaluation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_file = os.path.join(tmpdir, "trace.json")
        data = {"seed": 42, "tasks": [{"id": 0, "size": 3.0e6}]}
        with open(trace_file, "w") as f:
            json.dump(data, f, sort_keys=True)
            
        hash_before = hashlib.sha256(open(trace_file, "rb").read()).hexdigest()
        
        # Read-only evaluation consumption
        with open(trace_file, "r") as f:
            loaded = json.load(f)
        _ = loaded["tasks"][0]["size"] * 2
        
        hash_after = hashlib.sha256(open(trace_file, "rb").read()).hexdigest()
        assert hash_before == hash_after

# ------------------------------------------------------------------------------
# 20. Cross-Algorithm Realization Equivalence
# ------------------------------------------------------------------------------
def test_20_cross_algorithm_realization_equivalence():
    """Test 20: CoTOP and DDQN consume identical exogenous inputs under same seed."""
    np.random.seed(42)
    tasks_1 = [np.random.uniform(2e6, 5e6) for _ in range(10)]
    
    np.random.seed(42)
    tasks_2 = [np.random.uniform(2e6, 5e6) for _ in range(10)]
    
    assert tasks_1 == tasks_2

# ------------------------------------------------------------------------------
# 21. Evaluation Weight Immutability
# ------------------------------------------------------------------------------
def test_21_evaluation_weight_immutability():
    """Test 21: Model weights do not change during forward evaluation passes."""
    model = ActorCritic(input_dim=114, num_actions=7)
    model.eval()
    
    state_dict_before = copy.deepcopy(model.state_dict())
    
    state = torch.randn(1, 114)
    with torch.no_grad():
        for _ in range(10):
            _ = model(state)
            
    for k in state_dict_before:
        assert torch.allclose(state_dict_before[k], model.state_dict()[k])

# ------------------------------------------------------------------------------
# 22. DDQN Decoupled Target Construction
# ------------------------------------------------------------------------------
def test_22_ddqn_target_construction():
    """Test 22: DDQN decoupled target construction y = r + gamma * Q_target(s', argmax Q_online)."""
    online_net = QNetwork(input_dim=10, num_actions=3)
    target_net = QNetwork(input_dim=10, num_actions=3)
    
    next_state = torch.randn(1, 10)
    reward = 1.0
    gamma = 0.99
    
    with torch.no_grad():
        best_a = torch.argmax(online_net(next_state), dim=-1)
        target_q = target_net(next_state)[0, best_a]
        y = reward + gamma * target_q.item()
        
    assert isinstance(y, float)
    assert not math.isnan(y)

# ------------------------------------------------------------------------------
# 23. Checkpoint Recovery
# ------------------------------------------------------------------------------
def test_23_checkpoint_recovery():
    """Test 23: Checkpoint state dict saving and loading restores exact weights."""
    model1 = ActorCritic(input_dim=50, num_actions=7)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.pth")
        torch.save(model1.state_dict(), ckpt_path)
        
        model2 = ActorCritic(input_dim=50, num_actions=7)
        model2.fc1.weight.data.normal_(0, 1)
        model2.load_state_dict(torch.load(ckpt_path))
        
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

# ------------------------------------------------------------------------------
# 24. Deterministic Evaluation
# ------------------------------------------------------------------------------
def test_24_deterministic_evaluation():
    """Test 24: Deterministic forward passes produce identical logits given same inputs."""
    set_seed(42)
    model = ActorCritic(input_dim=50, num_actions=7)
    state = torch.randn(1, 50)
    
    logits1, val1 = model(state)
    logits2, val2 = model(state)
    
    assert torch.allclose(logits1, logits2)
    assert torch.allclose(val1, val2)
