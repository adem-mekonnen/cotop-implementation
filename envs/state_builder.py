import numpy as np
from typing import List
from envs.entities import Vehicle, Task, RSU

def build_state(vehicle: Vehicle, tasks: List[Task], rsus: List[RSU], config=None) -> np.ndarray:
    """
    Implements Eq 24: Constructs the state vector s(t) with fixed dimensions.
    
    Dimensions:
    - Vehicle: 4
    - Tasks: 4 * max_tasks
    - RSUs: 5 * num_rsus
    """
    # 1. Determine Dimensions from Config (Critical for preventing size mismatch)
    # We use these to pad the arrays so the length is ALWAYS the same.
    max_tasks = config.num_tasks_per_vehicle if config else len(tasks)
    num_rsus = config.num_rsus if config else len(rsus)
    
    expected_dim = 4 + (max_tasks * 4) + (num_rsus * 5)

    # 2. Handle NoneType Vehicle (Simulation start)
    if vehicle is None:
        return np.zeros(expected_dim, dtype=np.float32)

    state_elements = []
    
    # 3. Vehicle State (4 dims)
    state_elements.extend([
        vehicle.pos[0], 
        vehicle.pos[1], 
        vehicle.speed, 
        vehicle.dwell_time_T_stay
    ])
    
    # 4. Task Details (Fixed 4 * max_tasks dims)
    # If there are fewer tasks than max_tasks, we pad with 0s
    for i in range(max_tasks):
        if i < len(tasks):
            t = tasks[i]
            state_elements.extend([t.size_rho, t.cpu_phi, t.max_delay_d, t.priority])
        else:
            state_elements.extend([0.0, 0.0, 0.0, 0.0])
        
    # 5. RSU Status (Fixed 5 * num_rsus dims)
    for i in range(num_rsus):
        if i < len(rsus):
            r = rsus[i]
            state_elements.extend([
                r.location[0], 
                r.location[1], 
                r.cpu_capacity_f, 
                r.queue_length, 
                r.transmission_power_P_R
            ])
        else:
            state_elements.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                               
    # Final check: Ensure we didn't make a math error
    final_state = np.array(state_elements, dtype=np.float32)
    
    # If for some reason the size is wrong, force it to the expected dim
    if final_state.shape[0] != expected_dim:
        # This is a fallback to prevent the Thread Crash
        padded_state = np.zeros(expected_dim, dtype=np.float32)
        safe_len = min(final_state.shape[0], expected_dim)
        padded_state[:safe_len] = final_state[:safe_len]
        return padded_state

    return final_state