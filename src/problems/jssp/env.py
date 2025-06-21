import numpy as np
import pandas as pd
from src.problems.base.env import BaseEnv
from src.problems.jssp.components import Solution

class Env(BaseEnv):
    """JSSP env that stores the instance data, current solution, and problem state to support algorithm."""

    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "jssp")
        self.construction_steps = self.instance_data["job_num"] * self.instance_data["machine_num"]
        self.key_item = "current_makespan"
        self.compare = lambda x, y: y - x

    @property
    def is_complete_solution(self) -> bool:
        self.get_key_value()
        return self.unfinished_jobs == []

    def load_data(self, data_path: str) -> tuple:
        with open(data_path, "r") as file:
            lines = file.readlines()
            job_num = int(lines[0].split()[0])
            machine_num = int(lines[0].split()[1])
            job_operation_sequence = []
            job_operation_time = []
            for line in lines[1:]:
                data = line.strip().split()
                job_operation_sequence.append([int(data[i]) for i in range(0, len(data), 2)])
                job_operation_time.append([int(data[i + 1]) for i in range(0, len(data), 2)])
        return {"job_num": job_num, "machine_num": machine_num, "job_operation_sequence": np.array(job_operation_sequence), "job_operation_time": np.array(job_operation_time)}

    def init_solution(self) -> Solution:
        return Solution(job_sequences=[[] for _ in range(self.instance_data["machine_num"])], job_operation_sequence=self.instance_data["job_operation_sequence"], job_operation_index=[0] * self.instance_data["job_num"])

    def get_key_value(self, solution: Solution=None) -> float:
        """Get the key value of the current solution based on the key item."""
        if solution is None:
            solution = self.current_solution
        # Initialize dynamic state data
        job_operation_index = [0] * self.instance_data["job_num"]
        job_last_operation_end_times = [0] * self.instance_data["job_num"]
        machine_operation_index = [0] * self.instance_data["machine_num"]
        machine_last_operation_end_times = [0] * self.instance_data["machine_num"]
        current_operations = sum([len(job_sequence) for job_sequence in solution.job_sequences])

        for _ in range(current_operations):
            target_job_id = None
            target_machine_id = None
            target_operation_index = None
            for machine_id in range(self.instance_data["machine_num"]):
                machine_job_operation_index = machine_operation_index[machine_id]
                if machine_job_operation_index < len(solution.job_sequences[machine_id]):
                    job_id = solution.job_sequences[machine_id][machine_job_operation_index]
                    job_machine_operation_index = job_operation_index[job_id]
                    job_machine_id = self.instance_data["job_operation_sequence"][job_id, job_machine_operation_index]
                    if job_machine_id == machine_id:
                        target_job_id = job_id
                        target_machine_id = machine_id
                        target_operation_index = job_machine_operation_index
                        break
            if target_job_id is None:
                return None

            start_time = max(job_last_operation_end_times[target_job_id], machine_last_operation_end_times[target_machine_id])
            end_time = start_time + self.instance_data["job_operation_time"][target_job_id, target_operation_index]
            job_last_operation_end_times[target_job_id] = end_time
            machine_last_operation_end_times[target_machine_id] = end_time
            job_operation_index[target_job_id] += 1
            machine_operation_index[target_machine_id] += 1

        current_makespan = max(machine_last_operation_end_times)
        self.unfinished_jobs = []
        for job_id in range(self.instance_data["job_num"]):
            if job_operation_index[job_id] != len(self.instance_data["job_operation_sequence"][job_id]):
                self.unfinished_jobs.append(job_id)
        return current_makespan

    def validation_solution(self, solution: Solution=None) -> bool:
        """Check the validation of this solution in the following items:
            1. Exist operation: The operation should be valid
            2. Non-repeat: Each machine processes one operation at a time
        """
        if solution is None:
            solution = self.current_solution

        if not isinstance(solution, Solution) or not isinstance(solution.job_sequences, list):
            return False

        for machine_id, machine_operations in enumerate(solution.job_sequences):
            for operation in machine_operations:
                # Check exist operation
                if operation < 0 or operation >= self.instance_data["job_num"]:
                    return False

            # Check non-repeat
            if len(machine_operations) != len(set(machine_operations)):
                return False

        return True

