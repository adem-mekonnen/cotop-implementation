import numpy as np
from typing import List
from envs.entities import Vehicle, Task, RSU

def build_state(vehicle: Vehicle, tasks: List[Task], rsus: List[RSU]) -> np.ndarray:
    """
    Implements Eq 24: Constructs the state vector s(t).
    
    The state contains:
    - Vehicle positions and mobility features
    - Task details (size, CPU requirement, max delay, priority)
    - RSU queue status and capacities
    
    Args:
        vehicle: The vehicle requesting offloading.
        tasks: The parallel tasks generated for the vehicle.
        rsus: List of available RSUs in the environment.
        
    Returns:
        A flattened numpy array representing the state s(t).
    """
    if vehicle is None:
        # Return a zero-vector if no vehicle exists
        obs_dim = 4 + (len(tasks) * 4) + (len(rsus) * 5)
        return np.zeros(obs_dim, dtype=np.float32)

    state_elements = []
    
    # 1. Vehicle State (Eq 24 parts)
    state_elements.extend([vehicle.pos[0], vehicle.pos[1], vehicle.speed, vehicle.dwell_time_T_stay])
    
    # 2. Task Details
    for task in tasks:
        state_elements.extend([task.size_rho, task.cpu_phi, task.max_delay_d, task.priority])
        
    # 3. RSU Status
    for rsu in rsus:
        state_elements.extend([rsu.location[0], rsu.location[1], 
                               rsu.cpu_capacity_f, rsu.queue_length, rsu.transmission_power_P_R])
                               
    return np.array(state_elements, dtype=np.float32)
