from src.problems.jssp.components import Solution, AdvanceOperator

def most_work_remaining_930e(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """Most Work Remaining heuristic for JSSP.
    Selects the unfinished job with the maximum remaining work (total processing time of remaining operations) and returns an AdvanceOperator for that job to proceed with the next operation in sequence.
    
    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "job_operation_time" (numpy.ndarray):  The time cost for each operation in each job.
            - "unfinished_jobs" (list[int]): List of all unfinished jobs.
            - "job_operation_index" (list[int]): The index of the next operation to be scheduled for each job.
        algorithm_data (dict): Contains data specific to this algorithm. Not used in this heuristic.
        **kwargs: Any additional hyperparameters. Not used in this heuristic.
    
    Returns:
        The AdvanceOperator to move the most work remaining job one step ahead in the sequence.
        An empty dict, since this heuristic does not update the algorithm_data.
    """
    # Determine the job with the most work remaining
    max_remaining_work = -1
    selected_job_id = None
    for job_id in problem_state['unfinished_jobs']:
        remaining_operations = problem_state['job_operation_time'][job_id][problem_state['job_operation_index'][job_id]:]
        remaining_work = sum(remaining_operations)
        if remaining_work > max_remaining_work:
            max_remaining_work = remaining_work
            selected_job_id = job_id
    
    # If no job is selected, return None
    if selected_job_id is None:
        return None, {}
    
    # Create and return the AdvanceOperator for the selected job
    advance_operator = AdvanceOperator(job_id=selected_job_id)
    return advance_operator, {}