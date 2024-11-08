from src.problems.jssp.components import Solution, AdvanceOperator

def first_come_first_served_6c4f(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AdvanceOperator, dict]:
    """Implement the First Come First Served (FCFS) heuristic for the JSSP.
    This heuristic schedules the jobs in the order they arrive, without considering their processing times or other characteristics.

    Args:
        global_data (dict): Not used in this heuristic.
        state_data (dict): The state dictionary containing the current state information. For this algorithm, we use:
            - "unfinished_jobs" (list[int]): List of all unfinished jobs.
        algorithm_data (dict): Not used in this heuristic.
        get_state_data_function (callable): Function to get the state data for a new solution.
        **kwargs: Additional hyperparameters (not used in this heuristic).

    Returns:
        AdvanceOperator: The operator that advances the first unfinished job.
        dict: Empty dictionary as the algorithm data is not updated.
    """
    
    # Check if there are any unfinished jobs. If not, return None.
    if not state_data["unfinished_jobs"]:
        return None, {}
    
    # Retrieve the first job from the list of unfinished jobs.
    job_id = state_data["unfinished_jobs"][0]
    
    # Create an AdvanceOperator to schedule the next operation for the first job.
    operator = AdvanceOperator(job_id=job_id)
    
    # No algorithm data is updated, so we return an empty dictionary.
    return operator, {}