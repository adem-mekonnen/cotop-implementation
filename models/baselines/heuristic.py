import numpy as np
from typing import List
from envs.entities import RSU

def local_only_policy(state: np.ndarray) -> int:
    """
    Local execution strategy (Section V-B).
    Always chooses action 0, meaning offload only to the current local RSU,
    with no collaboration or task splitting to neighboring RSUs.
    
    Args:
        state (np.ndarray): The current environment state s(t).
        
    Returns:
        int: Action index 0.
    """
    return 0

def greedy_queue_policy(state: np.ndarray, rsus: List[RSU]) -> int:
    """
    Greedy load-balancing strategy (Section V-B).
    Looks at the queue_length of all available RSUs and chooses the one 
    with the minimum current load.
    
    Args:
        state (np.ndarray): The current environment state s(t).
        rsus (List[RSU]): List of RSU objects available in the environment.
        
    Returns:
        int: Action index corresponding to the RSU with the shortest queue.
    """
    if not rsus:
        return 0 # Fallback if RSU list is somehow empty
        
    min_queue_length = float('inf')
    best_action = 0
    
    for i, rsu in enumerate(rsus):
        if rsu.queue_length < min_queue_length:
            min_queue_length = rsu.queue_length
            best_action = i
            
    return best_action
