import math
from typing import List, Optional
from envs.entities import Task


def compute_task_priority_paper(
    task: Task, dwell_time: float, alpha: float = 0.3, beta: float = 0.7
) -> float:
    """
    Paper-Literal Eq. 23:
    P_i = alpha * e^(-1/T_stay) + beta * (rho_n,i / d_n,i)
    
    Primary baseline reproduction reference.
    Note: Under Table III parameters (rho ~ 2-5 MB, d ~ 20-30 s), the beta term is ~ 10^5
    while the dwell term is in [0, 1].
    """
    if dwell_time <= 0:
        dwell_term = 0.0
    else:
        dwell_term = math.exp(-1.0 / dwell_time)

    if task.max_delay_d <= 0:
        size_delay_term = 0.0
    else:
        size_delay_term = task.size_rho / task.max_delay_d

    return float(alpha * dwell_term + beta * size_delay_term)


def compute_task_priority_normalized(
    task: Task,
    dwell_time: float,
    alpha: float = 0.3,
    beta: float = 0.7,
    rho_max: float = 5.0e6,
    d_min: float = 20.0,
) -> float:
    """
    Normalized Eq. 23 Candidate (Experimental Stabilization):
    P_i = alpha * e^(-1/T_stay) + beta * ((rho_n,i / rho_max) / (d_n,i / d_min))
    
    Scientific Classification: SCIENTIFIC AMBIGUITY / IMPLEMENTATION STABILIZATION.
    Brings the beta term into controlled numerical magnitude without artificial clipping.
    """
    if dwell_time <= 0:
        dwell_term = 0.0
    else:
        dwell_term = math.exp(-1.0 / dwell_time)

    if task.max_delay_d <= 0 or d_min <= 0 or rho_max <= 0:
        size_delay_term = 0.0
    else:
        norm_rho = task.size_rho / rho_max
        norm_d = task.max_delay_d / d_min
        size_delay_term = norm_rho / norm_d

    return float(alpha * dwell_term + beta * size_delay_term)


# Primary default mapping: Paper-literal formula
compute_task_priority = compute_task_priority_paper


def prioritize_tasks_paper(
    tasks: List[Task], dwell_time: float, alpha: float = 0.3, beta: float = 0.7
) -> List[Task]:
    """Sorts tasks by paper-literal priority (Eq. 23), highest priority first."""
    for task in tasks:
        task.priority = compute_task_priority_paper(task, dwell_time, alpha, beta)
    return sorted(tasks, key=lambda t: t.priority, reverse=True)


def prioritize_tasks_normalized(
    tasks: List[Task],
    dwell_time: float,
    alpha: float = 0.3,
    beta: float = 0.7,
    rho_max: float = 5.0e6,
    d_min: float = 20.0,
) -> List[Task]:
    """Sorts tasks by normalized candidate priority, highest priority first."""
    for task in tasks:
        task.priority = compute_task_priority_normalized(
            task, dwell_time, alpha, beta, rho_max, d_min
        )
    return sorted(tasks, key=lambda t: t.priority, reverse=True)


# Default priority sort
prioritize_tasks = prioritize_tasks_paper
calculate_priority_and_sort = prioritize_tasks