from src.problems.jssp.components import Solution, AdvanceOperator
import random

def random_6512(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AdvanceOperator, dict]:
    """This heuristic randomly selects an unfinished job and advances its next operation in the job's processing sequence.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "job_num" (int): The total number of jobs in the problem.
            - "machine_num" (int): The total number of machines in the problem, also as operation num.
            - "job_operation_sequence" (numpy.ndarray): A list of jobs where each job is a list of operations in target sequence.
            - "job_operation_time" (numpy.ndarray):  The time cost for each operation in the target job.
            - "total_processing_times" (list[int]): The total processing time for each machine across all jobs in machine id.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "finished_jobs" (list[int]): List of all finished jobs.
            - "unfinished_jobs" (list[int]): List of all unfinished jobs.
            - "job_operation_index" (list[int]): The index of the next operation to be scheduled for each job.
            - "job_last_operation_end_times" (list[int]): The end time of the last operation for each job in the current solution.
            - "job_states" (pd.DataFrame): DataFrame for job states, columns=["JobID", "IsFinished", "CompletedOperationsForMachinesInSequence(MachineID,StartTime,EndTime)", "RemainingOperationsForMachineInSequence(MachineID)", "EndTimeForLastOperation"]
            - "machine_last_operation_end_times" (list[int]): The end time of the last operation for each machine, also as the available time for next operation.
            - "machine_states" (pd.DataFrame): DataFrame for machine states, columns=["MachineID", "CompletedOperationsForJobsInSequence(JobID,StartTime,EndTime)", "EndTimeForLastOperation"]
            - "current_makespan" (int): The time cost for current operation jobs, also known as the current_makespan.

    Returns:
        AdvanceOperator: The operator to advance the next operation for a randomly selected unfinished job.
        dict: Updated algorithm data (empty in this case).
    """
    # Check if there are any unfinished jobs
    unfinished_jobs = state_data["unfinished_jobs"]
    if not unfinished_jobs:
        return None, {}

    # Randomly select an unfinished job
    selected_job = random.choice(unfinished_jobs)

    # Create an AdvanceOperator for the selected job
    operator = AdvanceOperator(job_id=selected_job)

    # Return the operator and an empty dictionary as no algorithm data is updated
    return operator, {}