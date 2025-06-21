from src.problems.jssp.components import Solution, AdvanceOperator

def first_come_first_served_6c4f(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """Implement the First Come First Served (FCFS) heuristic for the JSSP.
    This heuristic schedules the jobs in the order they arrive, without considering their processing times or other characteristics.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "unfinished_jobs" (list[int]): List of all unfinished jobs.
        algorithm_data (dict): Not used in this heuristic.
        **kwargs: Additional hyperparameters (not used in this heuristic).

    Returns:
        AdvanceOperator: The operator that advances the first unfinished job.
        dict: Empty dictionary as the algorithm data is not updated.
    """
    
    # Check if there are any unfinished jobs. If not, return None.
    if not problem_state["unfinished_jobs"]:
        return None, {}
    
    # Retrieve the first job from the list of unfinished jobs.
    job_id = problem_state["unfinished_jobs"][0]
    
    # Create an AdvanceOperator to schedule the next operation for the first job.
    operator = AdvanceOperator(job_id=job_id)
    
    # No algorithm data is updated, so we return an empty dictionary.
    return operator, {}