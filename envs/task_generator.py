task_gen_code = '''
import random
from typing import List
from envs.entities import Task, SimulationConfig


class TaskGenerator:
    def __init__(self, config: SimulationConfig):
        """
        Generates a set S of I parallel tasks per vehicle (Sec III-A),
        sampling all ranges directly from the verified Table III config.
        """
        self.config = config
        # Baseline task count per vehicle -- using the lower bound of the range
        # for a fixed-size observation space (I varies [20,40] in the paper,
        # but our fixed-dim state vector requires a constant task count)
        self.num_tasks_I = config.num_tasks_per_vehicle_range[0]
        self._task_counter = 0  # ensures globally unique task IDs

    def generate_tasks_for_vehicle(self, vehicle_id: str) -> List[Task]:
        tasks = []
        for i in range(self.num_tasks_I):
            self._task_counter += 1
            task_id = self._task_counter  # simple incrementing int, avoids string-parsing bugs

            size_rho = random.uniform(*self.config.task_size_range)          # bytes, per Table III
            max_delay_d = random.uniform(*self.config.task_deadline_range)   # seconds, per Table III

            # CPU demand in cycles, capped at max_task_cpu (Mcycles -> cycles)
            max_cycles = self.config.max_task_cpu * 1.0e6
            cpu_phi = random.uniform(1.0e6, max_cycles)

            task = Task(
                task_id=task_id,
                vehicle_id=vehicle_id,
                size_rho=size_rho,
                cpu_phi=cpu_phi,
                max_delay_d=max_delay_d,
                priority=0.0,
            )
            tasks.append(task)

        return tasks
'''
with open('envs/task_generator.py', 'w') as f:
    f.write(task_gen_code.strip())
print("task_generator.py corrected: reads Table III ranges from config, fixed units (bytes not MB), safe task_id generation.")