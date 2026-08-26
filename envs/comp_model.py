# envs/comp_model.py

def calculate_case1_standalone(
    task_size_rho: float, 
    task_cpu_phi: float, 
    w_v2r: float, 
    rsu_cpu_f: float, 
    power_v: float, 
    compute_power_rsu: float,
    t_wait: float = 0.0
) -> tuple[float, float]:
    """
    Case 1: Standalone RSU execution.
    Calculates Total Delay (T_{n,m,i}^{total}) and Energy (E_i^{total}).
    
    References:
    - Eq 3: Transmission delay from Vehicle to RSU.
    - Eq 4: Computation delay at RSU.
    - Eq 5-6: Total Delay.
    - Eq 11-12: Total Energy.
    """
    # Unit Strictness: Convert task_size_rho to bits only once (1 Byte = 8 bits)
    task_size_bits = task_size_rho * 8
    
    # Eq 3: Transmission delay
    t_trans = task_size_bits / w_v2r if w_v2r > 0 else float('inf')
    
    # Eq 4: Computation delay
    t_comp = task_cpu_phi / rsu_cpu_f if rsu_cpu_f > 0 else float('inf')
    
    # Eq 5-6: Total Delay with sequential delay logic
    total_delay = t_trans + t_comp + t_wait
    
    # Eq 11-12: Total Energy
    # Eq 11: Transmission energy  E_trans = P_V * t_trans
    energy_trans = power_v * t_trans
    # Eq 12: Computation energy   E_comp = time * compute_power_watts
    energy_comp = t_comp * compute_power_rsu
    total_energy = energy_trans + energy_comp
    
    return total_delay, total_energy


def calculate_case2_collaboration(
    task_size_rho: float, 
    task_cpu_phi: float, 
    w_v2r: float, 
    w_r2r: float,
    rsu1_cpu_f: float, 
    rsu2_cpu_f: float,
    t1_dwell_time: float,
    power_v: float, 
    tx_power_rsu1: float,
    compute_power_rsu1: float,
    compute_power_rsu2: float,
    t_wait: float = 0.0
) -> tuple[float, float]:
    """
    Case 2: RSU Collaboration execution (Section III-C2, Eq. 7-10).
    Calculates Total Delay and Energy using parallel execution logic.
    
    References:
    - Eq 7: phi_{n,m,i}^{rest} = phi_{n,i} - t1 * F_m^{RSU}
    - Eq 8: T_{m,m',i}^{ts} = rho_{n,m,i}^{rest} / w_{m,m'}^{R2R}
    - Eq 9: T_{n,m',i}^{pro_rest} = phi_{n,m,i}^{rest} / F_{m'}^{RSU}
    - Eq 10: T_{total} = T^{up} + max(t1, t2 + t3) + T_{m'}^{wait}
    - Eq 11-12: Energy consumption for parallel computation and R2R transmission.
    """
    # Unit Strictness: Convert task_size_rho to bits only once (1 Byte = 8 bits)
    task_size_bits = task_size_rho * 8
    
    # V2R Transmission Delay (Eq. 3)
    t_v2r = task_size_bits / w_v2r if w_v2r > 0 else float('inf')
    
    # Standalone execution time required on RSU 1 alone
    t_comp1_full = task_cpu_phi / rsu1_cpu_f if rsu1_cpu_f > 0 else float('inf')
    
    # Determine t1: Duration that RSU 1 computes before/during parallel handover
    # If dwell time is less than standalone compute time, RSU 1 computes until vehicle leaves
    # If dwell time is ample and collaboration is chosen, partition optimally for parallel speedup
    if t1_dwell_time < t_comp1_full:
        t1 = max(t1_dwell_time, 0.0)
    else:
        # Parallel load partitioning ratio: proportional to computing capacity
        part_ratio = rsu1_cpu_f / (rsu1_cpu_f + rsu2_cpu_f) if (rsu1_cpu_f + rsu2_cpu_f) > 0 else 0.5
        t1 = min(t1_dwell_time, part_ratio * t_comp1_full)
        
    cpu_processed_rsu1 = min(rsu1_cpu_f * t1, task_cpu_phi)
    
    # Eq 7: Remaining CPU cycles to be processed at RSU 2
    remaining_cpu_phi = max(task_cpu_phi - cpu_processed_rsu1, 0.0)
    
    if remaining_cpu_phi <= 0:
        # All processed locally
        return calculate_case1_standalone(
            task_size_rho, task_cpu_phi, w_v2r, rsu1_cpu_f,
            power_v, compute_power_rsu1, t_wait
        )
    
    # Proportion of remaining data to transmit via R2R
    proportion_remaining = remaining_cpu_phi / task_cpu_phi
    remaining_size_bits = task_size_bits * proportion_remaining
    
    # Eq 8: t2 - inter-RSU transfer delay
    t2_inter_rsu = remaining_size_bits / w_r2r if w_r2r > 0 else float('inf')
    
    # Eq 9: t3 - remaining computation delay at RSU 2
    t3_comp2 = remaining_cpu_phi / rsu2_cpu_f if rsu2_cpu_f > 0 else float('inf')
    
    # Section III-C2 & Fig 2: Parallel execution processing delay = max(t1, t2 + t3)
    processing_delay = max(t1, t2_inter_rsu + t3_comp2)
    
    # Eq 10: Total Delay = V2R Upload + Processing Delay + Secondary RSU Wait Time
    total_delay = t_v2r + processing_delay + t_wait
    
    # Eq 11-12: Energy components
    energy_trans_v2r = power_v * t_v2r                       # Eq 12: V2R transmission
    energy_comp1     = t1 * compute_power_rsu1               # Eq 11: RSU1 computation
    energy_trans_r2r = tx_power_rsu1 * t2_inter_rsu          # Eq 12: RSU1->RSU2 relay
    energy_comp2     = t3_comp2 * compute_power_rsu2         # Eq 11: RSU2 computation
    
    total_energy = energy_trans_v2r + energy_comp1 + energy_trans_r2r + energy_comp2
    
    return total_delay, total_energy
