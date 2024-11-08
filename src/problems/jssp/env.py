import numpy as np
import pandas as pd
from src.problems.base.env import BaseEnv
from src.problems.jssp.components import Solution

class Env(BaseEnv):
    """JSSP env that stores the static global data, current solution, dynamic state and provide necessary support to the algorithm."""

    def __init__(self, data_name: str, mode: str, **kwargs):
        super().__init__(data_name, mode, "jssp")
        self.job_num, self.machine_num, self.job_operation_sequence, self.job_operation_time = self.data
        self.construction_steps = self.job_num * self.machine_num
        self.key_item = "current_makespan"
        self.compare = lambda x, y: y - x

    @property
    def is_complete_solution(self) -> bool:
        return self.state_data["unfinished_jobs"] == []

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
        return job_num, machine_num, np.array(job_operation_sequence), np.array(job_operation_time)

    def init_solution(self) -> Solution:
        return Solution(job_sequences=[[] for _ in range(self.machine_num)], job_operation_sequence=self.job_operation_sequence, job_operation_index=[0] * self.job_num)

    def get_global_data(self) -> dict:
        """Retrieve the global static information data as a dictionary.

        Returns:
            dict: A dictionary containing the global static information data with:
                - "job_operation_sequence" (numpy.ndarray): A list of jobs where each job is a list of operations in target sequence.
                - "job_operation_time" (numpy.ndarray): The time cost for each operation in target job.
                - "job_num" (int): The total number of jobs in the problem.
                - "machine_num" (int): The total number of machines in the problem, also as operation num.
        """

        global_data_dict = {
            "job_num": self.job_num,
            "machine_num": self.machine_num,
            "job_operation_sequence": self.job_operation_sequence,
            "job_operation_time": self.job_operation_time,
            "total_operation_num": self.job_num * self.machine_num,
        }
        return global_data_dict

    def get_state_data(self, solution: Solution=None) -> dict:
        """Retrieve the current dynamic state data as a dictionary.

        Returns:
            dict: A dictionary containing the current dynamic state data with:
                - "current_solution" (Solution): An instance of the Solution class representing the current solution.
                - "finished_jobs" (list[int]): List of all finished jobs.
                - "unfinished_jobs" (list[int]): List of all unfinished jobs.
                - "job_operation_index" (list[int]): The index of the next operation to be scheduled for each job.
                - "job_last_operation_end_times" (list[int]): The end time of the last operation for each job in current solution.
                - "machine_operation_index" (list[int]): The index of the next operation to be scheduled for each machine.
                - "machine_last_operation_end_times" (list[int]): The end time of the last operation for each machine, also as the available time for next operation.
                - "finished_operation_num" (int): The number of finished operation.
                - "current_makespan" (int): The time cost for current operation jobs, also known as the current_makespan.
                - "validation_solution" (callable): def validation_solution(solution: Solution) -> bool: function to check whether new solution is valid.
        """
        if solution is None:
            solution = self.current_solution

        # Initialize dynamic state data
        job_operation_index = [0] * self.job_num
        job_last_operation_end_times = [0] * self.job_num
        machine_operation_index = [0] * self.machine_num
        machine_last_operation_end_times = [0] * self.machine_num
        current_operations = sum([len(job_sequence) for job_sequence in solution.job_sequences])

        for _ in range(current_operations):
            target_job_id = None
            target_machine_id = None
            target_operation_index = None
            for machine_id in range(self.machine_num):
                machine_job_operation_index = machine_operation_index[machine_id]
                if machine_job_operation_index < len(solution.job_sequences[machine_id]):
                    job_id = solution.job_sequences[machine_id][machine_job_operation_index]
                    job_machine_operation_index = job_operation_index[job_id]
                    job_machine_id = self.job_operation_sequence[job_id, job_machine_operation_index]
                    if job_machine_id == machine_id:
                        target_job_id = job_id
                        target_machine_id = machine_id
                        target_operation_index = job_machine_operation_index
                        break
            if target_job_id is None:
                return None

            start_time = max(job_last_operation_end_times[target_job_id], machine_last_operation_end_times[target_machine_id])
            end_time = start_time + self.job_operation_time[target_job_id, target_operation_index]
            job_last_operation_end_times[target_job_id] = end_time
            machine_last_operation_end_times[target_machine_id] = end_time
            job_operation_index[target_job_id] += 1
            machine_operation_index[target_machine_id] += 1

        finished_jobs = []
        unfinished_jobs = []
        for job_id in range(self.job_num):
            if job_operation_index[job_id] == len(self.job_operation_sequence[job_id]):
                finished_jobs.append(job_id)
            else:
                unfinished_jobs.append(job_id)

        # Calculate current_makespan as the maximum end time across all machines
        current_makespan = max(machine_last_operation_end_times)

        # Compile the state data dictionary
        state_data_dict = {
            "current_solution": solution,
            "finished_jobs": finished_jobs,
            "unfinished_jobs": unfinished_jobs,
            "job_operation_index": job_operation_index,
            "job_last_operation_end_times": job_last_operation_end_times,
            "machine_operation_index": machine_operation_index,
            "machine_last_operation_end_times": machine_last_operation_end_times,
            "finished_operation_num": current_operations,
            "current_makespan": current_makespan,
            "validation_solution": self.validation_solution
        }
        return state_data_dict

    def validation_solution(self, solution: Solution) -> bool:
        """Check the validation of this solution in the following items:
            1. Exist operation: The operation should be valid
            2. Non-repeat: Each machine processes one operation at a time
        """
        if not isinstance(solution, Solution) or not isinstance(solution.job_sequences, list):
            return False

        for machine_id, machine_operations in enumerate(solution.job_sequences):
            for operation in machine_operations:
                # Check exist operation
                if operation < 0 or operation >= self.job_num:
                    return False

            # Check non-repeat
            if len(machine_operations) != len(set(machine_operations)):
                return False

        return True

    def get_observation(self) -> dict:
        return {
            "current_makespan": self.state_data["current_makespan"],
            "Finished Job Num": len(self.state_data["finished_jobs"]),
            "Finished Operation Num": self.state_data["finished_operation_num"],
        }

    def dump_result(self) -> str:
        content_dict = {
            "job_num": self.job_num,
            "machine_num": self.machine_num,
            "current_makespan": self.state_data["current_makespan"],
        }
        content = super().dump_result(content_dict)
        return content