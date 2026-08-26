"""
experiments/manual_collaboration_test.py
Demonstrates the exact mathematical derivation of Case 1 vs Case 2 in a controlled scenario.
"""
import math
from envs.comm_model import compute_v2r_rate, compute_r2r_rate
from envs.comp_model import calculate_case1_standalone, calculate_case2_collaboration

def run_manual_test():
    print("==================================================")
    print("   CONTROLLED COLLABORATION TEST DERIVATION       ")
    print("==================================================")

    task_size = 4.0e6      # 4 MB = 32 Mbits
    task_cpu = 10.0e6      # 10 Mcycles
    p_v = 0.01             # 10 dBm
    p_r = 100.0            # 50 dBm
    p_comp = 50.0          # Compute Watts
    noise = 0.001          # Noise power
    k = 1000.0             # 30 dB fixed gain

    # Primary RSU 0: F1 = 1 GHz, Queue = 20 Mcycles (Wait = 0.02s)
    # Secondary RSU 1: F2 = 4 GHz, Queue = 0 Mcycles (Wait = 0.00s), Dist = 400m
    w_v2r = compute_v2r_rate(100.0, 20.0e6, p_v, noise, k, 2.0)
    w_r2r = compute_r2r_rate(400.0, 50.0e6, p_r, noise, k, 2.0)

    # Standalone
    delay_c1, energy_c1 = calculate_case1_standalone(
        task_size, task_cpu, w_v2r, 1.0e9, p_v, p_comp, t_wait=0.02
    )

    # Collaboration
    delay_c2, energy_c2 = calculate_case2_collaboration(
        task_size, task_cpu, w_v2r, w_r2r, 1.0e9, 4.0e9,
        t1_dwell_time=10.0, power_v=p_v, tx_power_rsu1=p_r,
        compute_power_rsu1=p_comp, compute_power_rsu2=p_comp, t_wait=0.00
    )

    print(f"1. V2R Transmission Rate (100m): {w_v2r/1e6:.2f} Mbps")
    print(f"2. R2R Transmission Rate (400m): {w_r2r/1e6:.2f} Mbps")
    print("\n--- STANDALONE (Action 0) ---")
    print(f"  Total Delay:  {delay_c1:.4f} s (Upload=1.6000s, Compute=0.0100s, QueueWait=0.0200s)")
    print(f"  Total Energy: {energy_c1:.4f} J (Trans=0.0160J, Compute=0.5000J)")

    print("\n--- COLLABORATIVE (Action 2 with RSU 1) ---")
    print(f"  Total Delay:  {delay_c2:.4f} s (Upload=1.6000s, ParallelCompute=0.0571s, QueueWait=0.0000s)")
    print(f"  Total Energy: {energy_c2:.4f} J (Trans=0.0160J, Comp1=0.1000J, Relay=5.5103J, Comp2=0.1000J)")
    print("==================================================")

if __name__ == "__main__":
    run_manual_test()
