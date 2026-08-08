import random
from typing import List
from envs.entities import Task

class TaskGenerator:
    def __init__(self, num_tasks_I: int):
        """
        Generates a set S of I parallel tasks per vehicle (Sec III-A).
        """
        self.num_tasks_I = num_tasks_I

    def generate_tasks_for_vehicle(self, vehicle_id: int) -> List[Task]:
        """
        Generates parallel tasks with sizes between 2-5 MB (Table III).
        """
        tasks = []
        for i in range(self.num_tasks_I):
            task_id = int(f"{vehicle_id}{i}") # Simple unique task ID
            
            # Task size rho in MB (2 to 5 MB per Table III)
            size_rho = random.uniform(2.0, 5.0) 
            
            # Placeholders for cpu_phi and max_delay_d (can be adjusted based on Table III)
            cpu_phi = random.uniform(10.0, 50.0)   # Example CPU workload
            max_delay_d = random.uniform(0.1, 1.0) # Example max tolerable delay
            
            # Priority defaults to 0, will be calculated later by utils/task_priority.py
            priority = 0.0 
            
            task = Task(
                task_id=task_id,
                vehicle_id=vehicle_id,
                size_rho=size_rho,
                cpu_phi=cpu_phi,
                max_delay_d=max_delay_d,
                priority=priority
            )
            tasks.append(task)
            
        return tasks
