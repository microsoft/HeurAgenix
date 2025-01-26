from src.problems.jssp.components import Solution, AdvanceOperator

def longest_job_next_2e23(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AdvanceOperator, dict]:
    """Longest Job Next heuristic for JSSP.
    
    Prioritizes the job with the longest total processing time remaining. It uses the
    AdvanceOperator to schedule the next operation for the job with the maximum
    remaining processing time.

    Args:
        global_data (dict): Contains global data about the problem.
            - "job_operation_time" (numpy.ndarray):  The time cost for each operation in each job.
            - "job_num" (int): The total number of jobs in the problem.
            
        state_data (dict): Contains the current state of the solution.
            - "unfinished_jobs" (list[int]): List of all unfinished jobs.
            - "job_operation_index" (list[int]): The index of the next operation to be scheduled for each job.
            - "current_solution" (Solution): The current state of the job sequences on each machine.
            
        algorithm_data (dict): Contains data necessary for this algorithm.
            (No specific data needed for this algorithm; can be omitted or passed as an empty dict)
            
        get_state_data_function (callable): Function that takes a Solution instance and returns its state data.
        
        **kwargs: Additional hyperparameters (not used in this algorithm).

    Returns:
        (AdvanceOperator): The operator to advance the job with the longest processing time remaining.
        (dict): Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data from the global and state dictionaries.
    job_operation_time = global_data["job_operation_time"]
    unfinished_jobs = state_data["unfinished_jobs"]
    job_operation_index = state_data["job_operation_index"]
    
    # Check if there are any unfinished jobs. If not, return None.
    if not unfinished_jobs:
        return None, {}
    
    # Calculate remaining processing time for each unfinished job.
    remaining_times = {
        job_id: sum(job_operation_time[job_id][index:])
        for job_id, index in enumerate(job_operation_index) if job_id in unfinished_jobs
    }
    
    # Find the job with the maximum remaining processing time.
    job_id_to_schedule = max(remaining_times, key=remaining_times.get)
    
    # Create and return the AdvanceOperator for the job with the longest job next.
    advance_op = AdvanceOperator(job_id=job_id_to_schedule)
    
    return advance_op, {}