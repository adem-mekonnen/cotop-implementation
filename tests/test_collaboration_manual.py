"""
tests/test_collaboration_manual.py
Tests the exact hand-calculated controlled scenario comparing Case 1 vs Case 2.
"""
import pytest
import math
from envs.comm_model import compute_v2r_rate, compute_r2r_rate
from envs.comp_model import calculate_case1_standalone, calculate_case2_collaboration

def test_controlled_collaboration_hand_calculation():
    # Scenario setup:
    # Vehicle at 100m, Primary RSU at 0m (F1 = 1 GHz, Queue = 20 Mcycles)
    # Secondary RSU at 400m (F2 = 4 GHz, Queue = 0 Mcycles)
    # Task size: 4 MB (32 Mbits), CPU demand: 10 Mcycles
    task_size = 4.0e6
    task_cpu = 10.0e6
    p_v = 0.01
    p_r = 100.0
    p_comp = 50.0
    noise = 0.001
    k = 1000.0
    
    # 1. Rates
    # V2R at 100m: SINR = (0.01 * 1000) / (0.001 * 10000) = 1.0 -> Rate = 20 MHz * log2(2) = 20 Mbps
    w_v2r = compute_v2r_rate(100.0, 20.0e6, p_v, noise, k, 2.0)
    assert math.isclose(w_v2r, 20.0e6, rel_tol=1e-5)
    
    # R2R at 400m: SINR = (100.0 * 1000) / (0.001 * 160000) = 625.0 -> Rate = 50 MHz * log2(626)
    w_r2r = compute_r2r_rate(400.0, 50.0e6, p_r, noise, k, 2.0)
    expected_w_r2r = 50.0e6 * math.log2(626.0)
    assert math.isclose(w_r2r, expected_w_r2r, rel_tol=1e-5)

    # 2. Case 1 (Standalone on Primary)
    t_v2r = (4.0e6 * 8) / 20.0e6  # 1.6 s
    t_comp1 = 10.0e6 / 1.0e9       # 0.01 s
    t_wait1 = 20.0e6 / 1.0e9       # 0.02 s
    expected_delay_c1 = t_v2r + t_comp1 + t_wait1 # 1.63 s
    expected_energy_c1 = (p_v * t_v2r) + (t_comp1 * p_comp) # 0.016 + 0.5 = 0.516 J

    delay_c1, energy_c1 = calculate_case1_standalone(
        task_size, task_cpu, w_v2r, 1.0e9, p_v, p_comp, t_wait=t_wait1
    )
    assert math.isclose(delay_c1, expected_delay_c1, rel_tol=1e-5)
    assert math.isclose(energy_c1, expected_energy_c1, rel_tol=1e-5)

    # 3. Case 2 (Collaborative with Secondary RSU 2)
    # Dwell time ample, partition: F1/(F1+F2) = 1/5 -> phi1 = 2 Mcycles, phi_rest = 8 Mcycles
    t1 = 2.0e6 / 1.0e9 # 0.002 s
    rho_rest_bits = (4.0e6 * 8) * 0.8 # 25.6 Mbits
    t2 = rho_rest_bits / expected_w_r2r # 25.6e6 / 464.5e6 = 0.055113 s
    t3 = 8.0e6 / 4.0e9 # 0.002 s
    t_wait2 = 0.0 # 0 queue on secondary RSU
    expected_delay_c2 = t_v2r + max(t1, t2 + t3) + t_wait2
    expected_energy_c2 = (p_v * t_v2r) + (t1 * p_comp) + (p_r * t2) + (t3 * p_comp)

    delay_c2, energy_c2 = calculate_case2_collaboration(
        task_size, task_cpu, w_v2r, w_r2r, 1.0e9, 4.0e9,
        t1_dwell_time=10.0, power_v=p_v, tx_power_rsu1=p_r,
        compute_power_rsu1=p_comp, compute_power_rsu2=p_comp, t_wait=t_wait2
    )

    assert math.isclose(delay_c2, expected_delay_c2, rel_tol=1e-5)
    assert math.isclose(energy_c2, expected_energy_c2, rel_tol=1e-5)
    assert delay_c1 != delay_c2
    assert energy_c1 != energy_c2
