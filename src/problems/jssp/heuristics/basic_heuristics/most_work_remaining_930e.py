from src.problems.jssp.components import Solution, AdvanceOperator

def most_work_remaining_930e(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """
    Most-Work-Remaining (MWR) dispatch rule for JSSP. At each decision point it scans all unfinished jobs, sums the processing times of their unscheduled operations (job_operation_time[j][job_operation_index[j]:]), and selects the job with the largest remaining total. It then issues an AdvanceOperator for that job, appending its next operation to the queue of its required machine as defined by the solution’s job_operation_sequence and job_operation_index. This is a global argmax over unfinished jobs (not first-fit), advancing exactly one operation per call. The rule is machine-agnostic (no consideration of machine idleness, queue lengths, or blocking) and focuses on front-loading heavy jobs to mitigate long tails in makespan. Required inputs: job_operation_time, unfinished_jobs, job_operation_index. Time complexity: O(∑ lengths of remaining-operation slices across unfinished jobs); constant extra memory.
    
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