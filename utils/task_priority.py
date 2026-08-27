import math
from typing import List, Tuple, Any
from envs.entities import Task


def compute_task_priority(
    task: Task,
    dwell_time: float,
    alpha: float = 0.3,
    beta: float = 0.7,
    rho_max: float = 5.0e6,
    d_min: float = 20.0,
) -> float:
    """
    Eq. 23 with Normalized Terms:
    P_i = alpha * dwell_term + beta * size_delay_term

    where:
      dwell_term = exp(-1 / T_stay) if T_stay > 0 else 0.0 (bounded in (0, 1])
      size_delay_term = (task.size_rho / rho_max) / (task.max_delay_d / d_min)

    Reference normalization bounds from Table III:
      rho_max = 5.0e6 Bytes (upper bound of task_size_range [2.0e6, 5.0e6])
      d_min = 20.0 seconds (lower bound of task_deadline_range [20.0, 30.0])

    This ensures both terms are dimensionless and bounded in [0, 1],
    enabling alpha and beta to exert balanced, mathematically sound weighting.
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

    return alpha * dwell_term + beta * size_delay_term


def prioritize_tasks(
    tasks: List[Task],
    dwell_time: float,
    alpha: float = 0.3,
    beta: float = 0.7,
    rho_max: float = 5.0e6,
    d_min: float = 20.0,
) -> List[Task]:
    """
    Sorts tasks by normalized priority (Eq. 23), highest priority first.
    """
    for task in tasks:
        task.priority = compute_task_priority(task, dwell_time, alpha, beta, rho_max, d_min)

    return sorted(tasks, key=lambda t: t.priority, reverse=True)


def prioritize_task_queue(
    task_entries: List[Any],
    alpha: float = 0.3,
    beta: float = 0.7,
    rho_max: float = 5.0e6,
    d_min: float = 20.0,
) -> List[Any]:
    """
    Prioritizes a list of (vehicle, task) tuples across multiple concurrent vehicles.
    Uses each vehicle's specific dwell time T_stay.
    """
    for vehicle, task in task_entries:
        task.priority = compute_task_priority(
            task=task,
            dwell_time=getattr(vehicle, "dwell_time_T_stay", 0.0),
            alpha=alpha,
            beta=beta,
            rho_max=rho_max,
            d_min=d_min,
        )

    return sorted(task_entries, key=lambda entry: entry[1].priority, reverse=True)


# Backward-compatible alias
calculate_priority_and_sort = prioritize_tasks