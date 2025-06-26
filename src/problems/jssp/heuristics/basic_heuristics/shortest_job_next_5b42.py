from src.problems.jssp.components import AdvanceOperator

def shortest_job_next_5b42(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """Implements the Shortest Job Next heuristic for the Job Shop Scheduling Problem.
    This heuristic chooses the unfinished job with the shortest remaining processing time and advances its next operation.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "job_operation_time" (numpy.ndarray):  The time cost for each operation in each job.
            - "unfinished_jobs" (list[int]): List of all unfinished jobs.
            - "current_solution" (Solution): The current solution state.
            
        algorithm_data (dict): Contains data necessary for the algorithm's state, not used in this heuristic.

    Returns:
        AdvanceOperator: The operator to advance the next operation of the selected job.
        dict: An empty dictionary as this heuristic does not update algorithm data.
    """

    # Retrieve the necessary information from the global and state data
    job_operation_time = problem_state["job_operation_time"]
    unfinished_jobs = problem_state["unfinished_jobs"]
    job_operation_index = problem_state["current_solution"].job_operation_index

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