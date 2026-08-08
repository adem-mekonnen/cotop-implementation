# envs/comp_model.py

def calculate_case1_standalone(
    task_size_rho: float, 
    task_cpu_phi: float, 
    w_v2r: float, 
    rsu_cpu_f: float, 
    power_v: float, 
    power_rsu: float
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
    # Eq 3: Transmission delay
    t_trans = task_size_rho / w_v2r if w_v2r > 0 else float('inf')
    
    # Eq 4: Computation delay
    t_comp = task_cpu_phi / rsu_cpu_f if rsu_cpu_f > 0 else float('inf')
    
    # Eq 5-6: Total Delay (simplified queueing for now)
    total_delay = t_trans + t_comp
    
    # Eq 11-12: Total Energy (Transmission Energy + Computation Energy)
    energy_trans = power_v * t_trans
    energy_comp = power_rsu * t_comp
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
    power_rsu1: float,
    power_rsu2: float
) -> tuple[float, float]:
    """
    Case 2: RSU Collaboration execution.
    Calculates Total Delay and Energy using parallel execution logic.
    
    References:
    - Eq 7-10: Total Delay and components for collaboration.
    """
    # V2R Transmission Delay
    t_v2r = task_size_rho / w_v2r if w_v2r > 0 else float('inf')
    
    # RSU 1 computes part of the task during dwell time (t1)
    # The amount of CPU cycles RSU 1 can process during t1:
    cpu_processed_rsu1 = rsu1_cpu_f * t1_dwell_time
    
    if cpu_processed_rsu1 >= task_cpu_phi:
        # Task finishes within dwell time at RSU 1, falls back to Case 1 effectively
        return calculate_case1_standalone(task_size_rho, task_cpu_phi, w_v2r, rsu1_cpu_f, power_v, power_rsu1)
    
    # Remaining CPU cycles to be processed at RSU 2
    remaining_cpu_phi = task_cpu_phi - cpu_processed_rsu1
    
    # Assuming remaining data size is proportional to remaining CPU cycles
    proportion_remaining = remaining_cpu_phi / task_cpu_phi
    remaining_size_rho = task_size_rho * proportion_remaining
    
    # t2: inter-RSU transfer delay
    t2_inter_rsu = remaining_size_rho / w_r2r if w_r2r > 0 else float('inf')
    
    # t3: remaining computation delay at RSU 2
    t3_comp2 = remaining_cpu_phi / rsu2_cpu_f if rsu2_cpu_f > 0 else float('inf')
    
    # Crucial Logic (Fig 2 and Sec III-C-2): Parallel execution processing delay
    # Processing Delay = max(t1, t2 + t3)
    processing_delay = max(t1_dwell_time, t2_inter_rsu + t3_comp2)
    
    # Total Delay = V2R Transmission + Processing Delay
    total_delay = t_v2r + processing_delay
    
    # Energy components
    energy_trans_v2r = power_v * t_v2r
    energy_comp1 = power_rsu1 * t1_dwell_time
    energy_trans_r2r = power_rsu1 * t2_inter_rsu
    energy_comp2 = power_rsu2 * t3_comp2
    
    total_energy = energy_trans_v2r + energy_comp1 + energy_trans_r2r + energy_comp2
    
    return total_delay, total_energy
