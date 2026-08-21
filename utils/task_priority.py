task_priority_code = '''
import math
from typing import List
from envs.entities import Task


def compute_task_priority(task: Task, dwell_time: float, alpha: float = 0.3, beta: float = 0.7) -> float:
    """
    Eq. 23: P_i = alpha * e^(-1/T_stay) + beta * (rho_n,i / d_n,i)
    """
    if dwell_time <= 0:
        dwell_term = 0.0
    else:
        dwell_term = math.exp(-1.0 / dwell_time)

    if task.max_delay_d <= 0:
        size_delay_term = 0.0
    else:
        size_delay_term = task.size_rho / task.max_delay_d

    return alpha * dwell_term + beta * size_delay_term


def prioritize_tasks(tasks: List[Task], dwell_time: float, alpha: float = 0.3, beta: float = 0.7) -> List[Task]:
    """
    Sorts tasks by priority (Eq. 23), highest priority first.
    Function name matches what vec_env.py imports.
    """
    for task in tasks:
        task.priority = compute_task_priority(task, dwell_time, alpha, beta)

    return sorted(tasks, key=lambda t: t.priority, reverse=True)


# Backward-compatible alias, in case any other file still calls the old name
calculate_priority_and_sort = prioritize_tasks
'''
with open('utils/task_priority.py', 'w') as f:
    f.write(task_priority_code.strip())
print("task_priority.py corrected: function renamed to prioritize_tasks, formula now matches Eq. 23 exactly, backward-compat alias kept.")