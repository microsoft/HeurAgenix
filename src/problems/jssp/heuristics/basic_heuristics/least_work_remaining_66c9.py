from src.problems.jssp.components import Solution, AdvanceOperator

def least_work_remaining_66c9(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AdvanceOperator, dict]:
    """
    Selects the job with the least total processing time remaining from the unfinished jobs and returns an AdvanceOperator to schedule its next operation.

    Args:
        global_data (dict): The global data dict containing the global data. For this algorithm, we need:
            - "job_operation_time" (numpy.ndarray):  The time cost for each operation in target job.
        state_data (dict): The state dictionary containing the current state information. For this algorithm, we need:
            - "unfinished_jobs" (list[int]): List of all unfinished jobs.
            - "job_operation_index" (list[int]): The index of the next operation to be scheduled for each job.

    Returns:
        AdvanceOperator: Operator to advance the selected job's next operation.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary information from global_data
    job_operation_time = global_data["job_operation_time"]

    # Extract necessary state data
    unfinished_jobs = state_data["unfinished_jobs"]
    job_operation_index = state_data["job_operation_index"]

    # Initialize the least work remaining and corresponding job ID
    min_work_remaining = float('inf')
    job_to_advance = None

    # Iterate over unfinished jobs to find the one with the least work remaining
    for job_id in unfinished_jobs:
        remaining_time = sum(job_operation_time[job_id][job_operation_index[job_id]:])
        if remaining_time < min_work_remaining:
            min_work_remaining = remaining_time
            job_to_advance = job_id

    # If no job is found, return None, {}
    if job_to_advance is None:
        return None, {}

    # Create and return the AdvanceOperator for the job with the least work remaining
    return AdvanceOperator(job_to_advance), {}