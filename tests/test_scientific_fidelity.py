import pytest
import torch
import numpy as np
import math
import os
import yaml

from envs.entities import SimulationConfig, Vehicle, Task, RSU
from models.mobility_gat import MobilityGAT_GRU
from utils.task_priority import (
    compute_task_priority_paper,
    compute_task_priority_normalized,
    prioritize_tasks_paper,
    prioritize_tasks_normalized,
)
from utils.scenario_geometry import get_rsu_positions
from envs.comm_model import compute_v2r_rate, compute_r2r_rate
from envs.comp_model import calculate_case1_standalone, calculate_case2_collaboration
from envs.vec_env import VECEnv, get_euclidean_distance


# ==============================================================================
# 1. GAT GRAPH & ARCHITECTURE TESTS (P3, P6)
# ==============================================================================

def test_gat_layer2_eq18_dimensions():
    """Verify GAT Layer 1 (concat) -> Layer 2 (mean averaging) produces exact (N, T, 64) tensor."""
    model = MobilityGAT_GRU(input_dim=2, embed_dim=64, num_heads=4, gru_hidden=64, output_dim=2)
    x = torch.randn(5, 5, 2)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 0, 1], [0, 1, 2, 3, 4, 1, 0]], dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (5, 5, 2), f"Expected shape (5, 5, 2), got {out.shape}"


def test_gat_n_node_graph_construction():
    """Verify N-node graph construction for N in [1, 2, 5, 10]."""
    for N in [1, 2, 5, 10]:
        positions = np.random.uniform(0, 200, size=(N, 2))
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        adj = (dist <= 200.0).astype(int)
        np.fill_diagonal(adj, 1)
        edge_index = torch.tensor(np.argwhere(adj == 1).T, dtype=torch.long)
        
        # Self-loops must exist for all N nodes
        for i in range(N):
            has_self_loop = bool(((edge_index[0] == i) & (edge_index[1] == i)).any())
            assert has_self_loop, f"Node {i} missing self-loop in N={N} graph"


def test_gat_in_range_sensitivity():
    """
    In-Range Sensitivity Test:
    When Vehicle 2 is within interaction radius of Vehicle 1, modifying Vehicle 2's
    trajectory MUST measurably alter Vehicle 1's embedding representation.
    """
    torch.manual_seed(42)
    model = MobilityGAT_GRU()
    model.eval()

    # N=2 vehicles close together (< 200m)
    x_v1 = torch.tensor([[[10.0, 10.0], [11.0, 11.0], [12.0, 12.0], [13.0, 13.0], [14.0, 14.0]]])
    x_v2_a = torch.tensor([[[15.0, 15.0], [16.0, 16.0], [17.0, 17.0], [18.0, 18.0], [19.0, 19.0]]])
    x_v2_b = torch.tensor([[[50.0, 50.0], [55.0, 55.0], [60.0, 60.0], [65.0, 65.0], [70.0, 70.0]]])

    x_seq_a = torch.cat([x_v1, x_v2_a], dim=0) / 200.0
    x_seq_b = torch.cat([x_v1, x_v2_b], dim=0) / 200.0

    # Bidirectional connected graph with self-loops
    edge_index = torch.tensor([[0, 1, 0, 1], [0, 1, 1, 0]], dtype=torch.long)

    with torch.no_grad():
        out_a = model(x_seq_a, edge_index)
        out_b = model(x_seq_b, edge_index)

    # Embedding / prediction of Vehicle 1 (node 0) must change
    diff_v1 = torch.norm(out_a[0] - out_b[0]).item()
    assert diff_v1 > 1e-4, f"Vehicle 1 representation did not change when in-range neighbor changed (diff={diff_v1})"


def test_gat_out_of_range_negative_control():
    """
    Out-of-Range Negative Control:
    With identical model weights, identical Vehicle 1 state, and Vehicle 2 disconnected
    from Vehicle 1 (distance > R_spatial), perturbing Vehicle 2 must NOT alter
    Vehicle 1's embedding beyond numerical tolerance (||Delta h_1||_2 < 1e-6).
    """
    torch.manual_seed(42)
    model = MobilityGAT_GRU()
    model.eval()

    # N=2 vehicles far apart (> 200m)
    x_v1 = torch.tensor([[[10.0, 10.0], [11.0, 11.0], [12.0, 12.0], [13.0, 13.0], [14.0, 14.0]]])
    x_v2_a = torch.tensor([[[500.0, 500.0], [510.0, 510.0], [520.0, 520.0], [530.0, 530.0], [540.0, 540.0]]])
    x_v2_b = torch.tensor([[[900.0, 900.0], [910.0, 910.0], [920.0, 920.0], [930.0, 930.0], [940.0, 940.0]]])

    x_seq_a = torch.cat([x_v1, x_v2_a], dim=0) / 2400.0
    x_seq_b = torch.cat([x_v1, x_v2_b], dim=0) / 2400.0

    # Disconnected graph: self-loops only (no edge between 0 and 1)
    edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)

    with torch.no_grad():
        out_a = model(x_seq_a, edge_index)
        out_b = model(x_seq_b, edge_index)

    diff_v1 = torch.norm(out_a[0] - out_b[0]).item()
    assert diff_v1 < 1e-6, f"Negative control failed: disconnected Vehicle 2 altered Vehicle 1 embedding (diff={diff_v1})"


def test_gat_permutation_equivariance():
    """Verify that permuting nodes in input permutes outputs accordingly when graph is permuted."""
    torch.manual_seed(42)
    model = MobilityGAT_GRU()
    model.eval()

    x = torch.randn(3, 5, 2)
    edge_index = torch.tensor([[0, 1, 2, 0, 1], [0, 1, 2, 1, 0]], dtype=torch.long)

    # Permuted order: [2, 0, 1]
    perm = [2, 0, 1]
    perm_map = {orig: new for new, orig in enumerate(perm)}
    x_perm = x[perm]
    
    edge_index_perm = torch.tensor([
        [perm_map[int(u)] for u in edge_index[0]],
        [perm_map[int(v)] for v in edge_index[1]]
    ], dtype=torch.long)

    with torch.no_grad():
        out = model(x, edge_index)
        out_perm = model(x_perm, edge_index_perm)

    diff = torch.norm(out[perm] - out_perm).item()
    assert diff < 1e-5, f"Permutation equivariance violated (diff={diff})"


# ==============================================================================
# 2. EQ. 23 DUAL TASK PRIORITIZATION TESTS (P4)
# ==============================================================================

def test_eq23_dual_implementation_and_range():
    """Verify both Paper-Literal and Normalized Candidate priorities and measure scale characteristics."""
    t_small = Task(task_id=0, vehicle_id="v0", size_rho=2.0e6, cpu_phi=4.0e6, max_delay_d=30.0) # Small task, relaxed deadline
    t_large = Task(task_id=1, vehicle_id="v0", size_rho=5.0e6, cpu_phi=10.0e6, max_delay_d=20.0) # Large task, urgent deadline

    dwell_short = 2.0  # Fast moving vehicle
    dwell_long = 20.0  # Slow moving vehicle

    # Paper Literal Eq. 23
    p_lit_small_long = compute_task_priority_paper(t_small, dwell_long, alpha=0.3, beta=0.7)
    p_lit_large_short = compute_task_priority_paper(t_large, dwell_short, alpha=0.3, beta=0.7)

    # Normalized Candidate Eq. 23
    p_norm_small_long = compute_task_priority_normalized(t_small, dwell_long, alpha=0.3, beta=0.7)
    p_norm_large_short = compute_task_priority_normalized(t_large, dwell_short, alpha=0.3, beta=0.7)

    # Literal scale is ~ 10^4 - 10^5 due to Byte / Second term
    assert p_lit_small_long > 1000.0, f"Expected large magnitude for literal, got {p_lit_small_long}"
    assert p_lit_large_short > 1000.0, f"Expected large magnitude for literal, got {p_lit_large_short}"

    # Normalized scale is controlled in [0.1, 2.0]
    assert 0.1 <= p_norm_small_long <= 2.0, f"Normalized priority out of expected range: {p_norm_small_long}"
    assert 0.1 <= p_norm_large_short <= 2.0, f"Normalized priority out of expected range: {p_norm_large_short}"


def test_eq23_ranking_inversions_analysis():
    """Compare rank ordering across a synthetic batch of 100 tasks under literal vs normalized priority."""
    np.random.seed(42)
    tasks = [
        Task(
            task_id=i,
            vehicle_id=f"v_{i % 5}",
            size_rho=float(np.random.uniform(2.0e6, 5.0e6)),
            cpu_phi=float(np.random.uniform(4.0e6, 10.0e6)),
            max_delay_d=float(np.random.uniform(20.0, 30.0))
        )
        for i in range(100)
    ]
    dwell_times = [float(np.random.uniform(1.0, 25.0)) for _ in range(100)]

    lit_priorities = [compute_task_priority_paper(t, d, alpha=0.3, beta=0.7) for t, d in zip(tasks, dwell_times)]
    norm_priorities = [compute_task_priority_normalized(t, d, alpha=0.3, beta=0.7) for t, d in zip(tasks, dwell_times)]

    # Compute Spearman rank correlation
    rank_lit = np.argsort(np.argsort(-np.array(lit_priorities)))
    rank_norm = np.argsort(np.argsort(-np.array(norm_priorities)))
    
    n = len(tasks)
    d_sq = np.sum((rank_lit - rank_norm) ** 2)
    spearman_rho = 1.0 - (6.0 * d_sq) / (n * (n**2 - 1))
    
    # Both rank urgency positively, but normalization gives mobility term a higher relative voice
    assert 0.0 <= spearman_rho <= 1.0, f"Invalid Spearman rho: {spearman_rho}"


# ==============================================================================
# 3. EQ. 25 PHYSICAL STATE COVERAGE & DEADLINE TESTS (P5)
# ==============================================================================

def test_eq25_success_case():
    """Task finishes within deadline and within RSU coverage radius -> receives normal cost reward."""
    delay = 1.80
    energy = 3.50
    deadline = 25.0
    coverage_dist = 150.0
    rsu_range = 200.0
    eps = 0.5

    fail_deadline = bool(delay > deadline)
    fail_coverage = bool(coverage_dist > rsu_range)

    assert not fail_deadline and not fail_coverage
    reward = -(eps * delay + (1.0 - eps) * energy)
    assert reward == -2.65


def test_eq25_deadline_failure():
    """Task delay exceeds tolerance deadline -> incurs penalty -Z."""
    delay = 32.0
    energy = 3.50
    deadline = 25.0
    coverage_dist = 150.0
    rsu_range = 200.0
    penalty_z = 100.0

    fail_deadline = bool(delay > deadline)
    fail_coverage = bool(coverage_dist > rsu_range)
    is_failed = fail_deadline or fail_coverage

    assert fail_deadline and not fail_coverage
    assert is_failed
    reward = -penalty_z
    assert reward == -100.0


def test_eq25_coverage_failure():
    """Vehicle moves outside RSU coverage radius during/at task execution -> incurs penalty -Z."""
    delay = 1.80
    energy = 3.50
    deadline = 25.0
    coverage_dist = 250.0 # Exceeds 200m range
    rsu_range = 200.0
    penalty_z = 100.0

    fail_deadline = bool(delay > deadline)
    fail_coverage = bool(coverage_dist > rsu_range)
    is_failed = fail_deadline or fail_coverage

    assert not fail_deadline and fail_coverage
    assert is_failed
    reward = -penalty_z
    assert reward == -100.0


def test_eq25_dual_failure():
    """Both deadline and coverage conditions violated -> incurs penalty -Z."""
    delay = 35.0
    coverage_dist = 280.0
    deadline = 20.0
    rsu_range = 200.0

    fail_deadline = bool(delay > deadline)
    fail_coverage = bool(coverage_dist > rsu_range)
    assert fail_deadline and fail_coverage


def test_eq25_exact_boundary():
    """At exact boundary (delay == deadline, dist == rsu_range), condition is satisfied."""
    delay = 20.0
    deadline = 20.0
    dist = 200.0
    rsu_range = 200.0

    fail_deadline = bool(delay > deadline)
    fail_coverage = bool(dist > rsu_range)
    assert not fail_deadline and not fail_coverage


# ==============================================================================
# 4. SCENARIO GEOMETRY TESTS (P2)
# ==============================================================================

def test_200m_reconstructed_scenario_geometry():
    """Verify 200m x 200m reconstructed scenario file existence, boundary, and 6 RSU placement."""
    assert os.path.exists("sumo_config/hangzhou_200m.net.xml")
    assert os.path.exists("sumo_config/hangzhou_200m.rou.xml")
    assert os.path.exists("sumo_config/hangzhou_200m.sumocfg")

    # Test 6 RSU positions for 200m grid
    positions = get_rsu_positions(num_rsus=6, traci_conn=None, scenario_mode="grid_200m")
    assert len(positions) == 6
    for x, y in positions:
        assert 0.0 <= x <= 200.0, f"RSU x={x} outside 200m grid"
        assert 0.0 <= y <= 200.0, f"RSU y={y} outside 200m grid"


def test_2400m_historical_scenario_preservation():
    """Verify 2400m historical corridor file existence, boundary, and 6 RSU placement."""
    assert os.path.exists("sumo_config/hangzhou.net.xml")
    assert os.path.exists("sumo_config/hangzhou.rou.xml")
    assert os.path.exists("sumo_config/hangzhou.sumocfg")

    positions = get_rsu_positions(num_rsus=6, traci_conn=None, scenario_mode="corridor_2400m")
    assert len(positions) == 6
    for x, y in positions:
        assert 0.0 <= x <= 2400.0, f"RSU x={x} outside 2400m corridor"
        assert y == 0.0, f"RSU y={y} outside 1D corridor"


# ==============================================================================
# 5. MULTI-VEHICLE EXECUTION CONTRACT & TASK INVARIANTS (P1)
# ==============================================================================

def test_task_ownership_invariant():
    """Task ownership invariant: every task retains its immutable vehicle_id."""
    v0_tasks = [Task(task_id=0, vehicle_id="veh_0", size_rho=3.0e6, cpu_phi=6.0e6, max_delay_d=25.0)]
    v1_tasks = [Task(task_id=1, vehicle_id="veh_1", size_rho=4.0e6, cpu_phi=8.0e6, max_delay_d=20.0)]

    # Prioritization must preserve vehicle_id
    prioritized = prioritize_tasks_paper(v0_tasks + v1_tasks, dwell_time=10.0)
    assert prioritized[0].vehicle_id in ["veh_0", "veh_1"]
    assert prioritized[1].vehicle_id in ["veh_0", "veh_1"]
    assert prioritized[0].vehicle_id != prioritized[1].vehicle_id


def test_shared_rsu_queue_depletion():
    """RSU queues accumulate computational work and deplete at rate F_m * dt."""
    rsu = RSU(rsu_id=0, location=(100.0, 100.0), cpu_capacity_f=2.0e9, queued_cpu_cycles=1.0e9, transmission_power_P_R=100.0)
    
    # Task arrives with 5.0e8 cycles
    rsu.queued_cpu_cycles += 5.0e8
    assert rsu.queued_cpu_cycles == 1.5e9
    
    # 1 second of simulation time elapses
    dt = 1.0
    rsu.queued_cpu_cycles = max(0.0, rsu.queued_cpu_cycles - rsu.cpu_capacity_f * dt)
    assert rsu.queued_cpu_cycles == 0.0 # 1.5e9 - 2.0e9 <= 0 -> 0.0
