from typing import List
from envs.entities import Task

def calculate_priority_and_sort(
    tasks: List[Task], 
    vehicle_dwell_time: float, 
    alpha: float = 0.3, 
    beta: float = 0.7
) -> List[Task]:
    """
    Implements Equation 23 to calculate task priority P_i and sort the tasks.
    
    Args:
        tasks: List of parallel tasks to sort.
        vehicle_dwell_time: The dwell time T_stay of the vehicle at the current RSU.
        alpha: Weight parameter, default 0.3.
        beta: Weight parameter, default 0.7.
        
    Returns:
        List of tasks sorted by priority (highest priority first).
    """
    
    for task in tasks:
        # ---------------------------------------------------------------------
        # Eq 23 Formulation
        # Note: This is a generalized approximation of Eq 23 using alpha and beta.
        # P_i = alpha * (Delay Factor) + beta * (Size/Computation Factor)
        # Adjust the exact variables here based on the precise text of Equation 23.
        # ---------------------------------------------------------------------
        
        # Example component 1: How tight is the deadline relative to dwell time?
        delay_factor = vehicle_dwell_time / task.max_delay_d if task.max_delay_d > 0 else 0
        
        # Example component 2: The size of the task
        size_factor = task.size_rho 
        
        # Calculate P_i
        task.priority = (alpha * delay_factor) + (beta * size_factor)
        
    # Sort the tasks based on the calculated priority P_i in descending order
    # (Assuming higher P_i means higher priority for scheduling/offloading)
    sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
    
    return sorted_tasks
