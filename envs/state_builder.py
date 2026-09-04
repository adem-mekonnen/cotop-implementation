import numpy as np
from typing import List
from envs.entities import Vehicle, Task, RSU

def build_state(vehicle: Vehicle, tasks: List[Task], rsus: List[RSU], config=None) -> np.ndarray:
    """
    Implements Eq 24: Constructs the normalized state vector s(t) with fixed dimensions.
    
    Dimensions:
    - Vehicle: 4
    - Tasks: 4 * max_tasks
    - RSUs: 5 * num_rsus
    """
    # 1. Determine Dimensions from Config (Critical for preventing size mismatch)
    max_tasks = config.num_tasks_per_vehicle_range[0] if config else len(tasks)
    num_rsus = config.num_rsus if config else len(rsus)
    
    expected_dim = 4 + (max_tasks * 4) + (num_rsus * 5)

    # 2. Handle NoneType Vehicle (Simulation start)
    if vehicle is None:
        return np.zeros(expected_dim, dtype=np.float32)

    state_elements = []
    
    # 3. Vehicle State (4 dims)
    # Normalization factors
    map_length = getattr(config, 'map_scale', 2400.0) if config else 2400.0
    max_cpu = 4.0e9
    max_size = 5.0e6
    max_speed = config.vehicle_speed_range[1] if config else 40.0
    max_dwell = 100.0
    
    state_elements.extend([
        vehicle.pos[0] / map_length, 
        vehicle.pos[1] / map_length, 
        vehicle.speed / max_speed, 
        vehicle.dwell_time_T_stay / max_dwell
    ])
    
    # 4. Task Details (Fixed 4 * max_tasks dims)
    max_delay = config.task_deadline_range[1] if config else 30.0
    for i in range(max_tasks):
        if i < len(tasks):
            t = tasks[i]
            state_elements.extend([
                t.size_rho / max_size, 
                t.cpu_phi / max_cpu, 
                t.max_delay_d / max_delay, 
                t.priority
            ])
        else:
            state_elements.extend([0.0, 0.0, 0.0, 0.0])
        
    # 5. RSU Status (Fixed 5 * num_rsus dims)
    max_tx_power = config.tx_power_rsu if config else 100.0
    # Eq 5: N_queue is in CPU cycles. If we allow up to 100 tasks of max_task_cpu, that's max queue.
    max_queue_cycles = 100.0 * (config.max_task_cpu * 1e6 if config else 10.0e6)
    for i in range(num_rsus):
        if i < len(rsus):
            r = rsus[i]
            state_elements.extend([
                r.location[0] / map_length, 
                r.location[1] / map_length, 
                r.cpu_capacity_f / max_cpu, 
                r.queued_cpu_cycles / max_queue_cycles, 
                r.transmission_power_P_R / max_tx_power
            ])
        else:
            state_elements.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                               
    final_state = np.array(state_elements, dtype=np.float32)
    
    # If for some reason the size is wrong, force it to the expected dim
    if final_state.shape[0] != expected_dim:
        padded_state = np.zeros(expected_dim, dtype=np.float32)
        safe_len = min(final_state.shape[0], expected_dim)
        padded_state[:safe_len] = final_state[:safe_len]
        return padded_state

    return final_state