"""
sanity_check.py: Independent deterministic verification of CoTOP system model equations.
Compares exact closed-form analytical math against repository implementation functions.
"""
import math
import sys
from envs.comm_model import compute_v2r_rate, compute_r2r_rate
from envs.comp_model import calculate_case1_standalone, calculate_case2_collaboration
from utils.task_priority import compute_task_priority
from envs.entities import Task

def check_close(name, calculated, expected, tol=1e-5):
    diff = abs(calculated - expected)
    if diff <= tol:
        print(f"  [PASS] {name}: {calculated:.8f} == {expected:.8f} (diff: {diff:.2e})")
        return True
    else:
        print(f"  [FAIL] {name}: calculated {calculated:.8f} != expected {expected:.8f} (diff: {diff:.2e})")
        return False

def run_sanity_checks():
    all_passed = True
    print("==================================================")
    print("      CoTOP SYSTEM MODEL SANITY CHECK             ")
    print("==================================================")

    # -------------------------------------------------------------------------
    # 1. Communication Model (Eq. 1: V2R Shannon Capacity)
    # -------------------------------------------------------------------------
    print("\n1. Testing V2R Communication Model (Eq. 1):")
    # Parameters
    dist_v2r = 100.0           # meters
    bw_v2r = 20.0e6            # 20 MHz
    power_v = 0.01             # 10 dBm = 0.01 Watts
    noise_power = 0.001        # 0.001 dBm = 0.001 Watts
    fixed_loss_k = 1000.0      # 30 dB = 1000.0
    path_loss_factor = 2.0     # sigma = 2

    # Hand calculation:
    # SINR = (0.01 * 1000.0) / (0.001 * 100.0^2) = 10.0 / 10.0 = 1.0
    # Rate = 20e6 * log2(1 + 1.0) = 20,000,000 bps
    expected_sinr = 1.0
    expected_v2r_rate = bw_v2r * math.log2(1.0 + expected_sinr)
    calc_v2r_rate = compute_v2r_rate(
        distance=dist_v2r,
        bandwidth_B=bw_v2r,
        power_P_V=power_v,
        noise_power=noise_power,
        fixed_loss_k=fixed_loss_k,
        path_loss_factor=path_loss_factor
    )
    all_passed &= check_close("V2R Rate (bps)", calc_v2r_rate, expected_v2r_rate)

    # -------------------------------------------------------------------------
    # 2. Communication Model (Eq. 2: R2R Shannon Capacity)
    # -------------------------------------------------------------------------
    print("\n2. Testing R2R Communication Model (Eq. 2):")
    dist_r2r = 400.0           # meters
    bw_r2r = 50.0e6            # 50 MHz
    power_r = 100.0            # 50 dBm = 100 Watts
    
    # Hand calculation:
    # SINR = (100.0 * 1000.0) / (0.001 * 400.0^2) = 100000.0 / 160.0 = 625.0
    # Rate = 50e6 * log2(1 + 625.0) = 50e6 * log2(626.0)
    expected_r2r_sinr = 625.0
    expected_r2r_rate = bw_r2r * math.log2(1.0 + expected_r2r_sinr)
    calc_r2r_rate = compute_r2r_rate(
        distance=dist_r2r,
        bandwidth_B=bw_r2r,
        power_P_R=power_r,
        noise_power=noise_power,
        fixed_loss_k=fixed_loss_k,
        path_loss_factor=path_loss_factor
    )
    all_passed &= check_close("R2R Rate (bps)", calc_r2r_rate, expected_r2r_rate)

    # -------------------------------------------------------------------------
    # 3. Computation Model — Case 1: Standalone (Eq. 3, 4, 5, 6, 11, 12)
    # -------------------------------------------------------------------------
    print("\n3. Testing Case 1: Standalone Offloading (Eq. 3, 4, 5, 6, 11, 12):")
    task_size_bytes = 2.0e6    # 2 MB
    task_cpu_cycles = 10.0e6   # 10 Mcycles
    rsu_cpu_f = 1.0e9          # 1 GHz
    compute_power_rsu = 50.0   # 50 Watts
    t_wait = 0.0               # Queue wait time

    # Hand calculation:
    # task_size_bits = 2e6 * 8 = 16,000,000 bits
    # t_trans = 16e6 / 20e6 = 0.8 s
    # t_comp = 10e6 / 1e9 = 0.01 s
    # total_delay = 0.8 + 0.01 + 0.0 = 0.81 s
    # energy_trans = 0.01 * 0.8 = 0.008 J
    # energy_comp = 0.01 * 50.0 = 0.5 J
    # total_energy = 0.008 + 0.5 = 0.508 J
    expected_c1_delay = 0.81
    expected_c1_energy = 0.508
    calc_c1_delay, calc_c1_energy = calculate_case1_standalone(
        task_size_rho=task_size_bytes,
        task_cpu_phi=task_cpu_cycles,
        w_v2r=calc_v2r_rate,
        rsu_cpu_f=rsu_cpu_f,
        power_v=power_v,
        compute_power_rsu=compute_power_rsu,
        t_wait=t_wait
    )
    all_passed &= check_close("Case 1 Total Delay (s)", calc_c1_delay, expected_c1_delay)
    all_passed &= check_close("Case 1 Total Energy (J)", calc_c1_energy, expected_c1_energy)

    # -------------------------------------------------------------------------
    # 4. Computation Model — Case 2: Collaboration (Eq. 7, 8, 9, 10, 11, 12)
    # -------------------------------------------------------------------------
    print("\n4. Testing Case 2: Collaborative Offloading (Eq. 7, 8, 9, 10, 11, 12):")
    t1_dwell = 0.005           # 5 ms dwell time at RSU 1
    rsu2_cpu_f = 2.0e9         # 2 GHz at RSU 2
    compute_power_rsu2 = 50.0  # 50 Watts

    # Hand calculation:
    # t_v2r = 0.8 s
    # RSU 1 processes: 1e9 * 0.005 = 5e6 cycles
    # Remaining cycles = 10e6 - 5e6 = 5e6 cycles (50%)
    # Remaining size bits = 16e6 * 0.5 = 8e6 bits
    # t2_r2r = 8e6 / calc_r2r_rate
    # t3_comp2 = 5e6 / 2e9 = 0.0025 s
    # processing_delay = max(t1_dwell, t2_r2r + t3_comp2)
    # total_delay = 0.8 + processing_delay + 0.0
    # energy_trans_v2r = 0.01 * 0.8 = 0.008 J
    # energy_comp1 = 0.005 * 50.0 = 0.25 J
    # energy_trans_r2r = 100.0 * t2_r2r
    # energy_comp2 = 0.0025 * 50.0 = 0.125 J
    # total_energy = 0.008 + 0.25 + energy_trans_r2r + 0.125
    expected_t2_r2r = (8.0e6) / calc_r2r_rate
    expected_t3 = (5.0e6) / rsu2_cpu_f
    expected_c2_delay = 0.8 + max(t1_dwell, expected_t2_r2r + expected_t3)
    expected_c2_energy = 0.008 + (0.005 * 50.0) + (power_r * expected_t2_r2r) + (expected_t3 * 50.0)

    calc_c2_delay, calc_c2_energy = calculate_case2_collaboration(
        task_size_rho=task_size_bytes,
        task_cpu_phi=task_cpu_cycles,
        w_v2r=calc_v2r_rate,
        w_r2r=calc_r2r_rate,
        rsu1_cpu_f=rsu_cpu_f,
        rsu2_cpu_f=rsu2_cpu_f,
        t1_dwell_time=t1_dwell,
        power_v=power_v,
        tx_power_rsu1=power_r,
        compute_power_rsu1=compute_power_rsu,
        compute_power_rsu2=compute_power_rsu2,
        t_wait=t_wait
    )
    all_passed &= check_close("Case 2 Total Delay (s)", calc_c2_delay, expected_c2_delay)
    all_passed &= check_close("Case 2 Total Energy (J)", calc_c2_energy, expected_c2_energy)

    # -------------------------------------------------------------------------
    # 5. Task Priority (Eq. 23)
    # -------------------------------------------------------------------------
    print("\n5. Testing Task Priority Formula (Eq. 23):")
    alpha = 0.3
    beta = 0.7
    t_stay = 10.0
    task = Task(task_id=1, vehicle_id="v1", size_rho=2.0e6, cpu_phi=10.0e6, max_delay_d=25.0)
    # Hand calculation:
    # P_i = alpha * exp(-1 / T_stay) + beta * (size_rho / max_delay_d)
    # P_i = 0.3 * exp(-0.1) + 0.7 * (2.0e6 / 25.0) = 0.3 * 0.904837418 + 0.7 * 80000 = 0.271451225 + 56000.0 = 56000.2714512
    expected_priority = alpha * math.exp(-1.0 / t_stay) + beta * (task.size_rho / task.max_delay_d)
    calc_priority = compute_task_priority(task, dwell_time=t_stay, alpha=alpha, beta=beta)
    all_passed &= check_close("Task Priority", calc_priority, expected_priority)

    print("\n==================================================")
    if all_passed:
        print("  >>> ALL SYSTEM MODEL SANITY CHECKS PASSED <<<   ")
        print("==================================================")
        return 0
    else:
        print("  >>> SOME CHECKS FAILED! REVIEW EQUATIONS! <<<   ")
        print("==================================================")
        return 1

if __name__ == "__main__":
    sys.exit(run_sanity_checks())
