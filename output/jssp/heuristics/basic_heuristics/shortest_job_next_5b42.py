from src.problems.jssp.components import AdvanceOperator

def shortest_job_next_5b42(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AdvanceOperator, dict]:
    """Implements the Shortest Job Next heuristic for the Job Shop Scheduling Problem.
    This heuristic chooses the unfinished job with the shortest remaining processing time and advances its next operation.

    Args:
        global_data (dict): Contains global static information about the problem.
            - "job_operation_time" (numpy.ndarray):  The time cost for each operation in each job.
            
        state_data (dict): Contains the current state information of the solution.
            - "unfinished_jobs" (list[int]): List of all unfinished jobs.
            - "current_solution" (Solution): The current solution state.
            
        algorithm_data (dict): Contains data necessary for the algorithm's state, not used in this heuristic.
        get_state_data_function (callable): A function to get the state data for a given solution.

    Returns:
        AdvanceOperator: The operator to advance the next operation of the selected job.
        dict: An empty dictionary as this heuristic does not update algorithm data.
    """

    # Retrieve the necessary information from the global and state data
    job_operation_time = global_data["job_operation_time"]
    unfinished_jobs = state_data["unfinished_jobs"]
    job_operation_index = state_data["current_solution"].job_operation_index

    # Check if there are any unfinished jobs
    if not unfinished_jobs:
        return None, {}

    # Calculate the total remaining processing time for each unfinished job
    remaining_times = {job: sum(times[job_operation_index[job]:]) 
                       for job, times in enumerate(job_operation_time) if job in unfinished_jobs}

    # Find the job with the shortest total remaining processing time
    shortest_job = min(remaining_times, key=remaining_times.get)

    # Generate the operator to advance the selected job
    operator = AdvanceOperator(job_id=shortest_job)

    return operator, {}