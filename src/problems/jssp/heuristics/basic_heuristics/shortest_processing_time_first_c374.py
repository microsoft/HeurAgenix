from src.problems.jssp.components import Solution, AdvanceOperator

def shortest_processing_time_first_c374(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """Implements the Shortest Processing Time first heuristic for the JSSP.
    
    This heuristic selects the unfinished job with the shortest next operation processing time and uses an AdvanceOperator to schedule it on the corresponding machine.
    
    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "job_operation_time" (numpy.ndarray):  The time cost for each operation in each job.
            - "unfinished_jobs" (list[int]): List of all unfinished jobs.
            - "current_solution" (Solution): The current solution state.
            - "job_operation_index" (list[int]): The index of the next operation to be scheduled for each job.
            
        algorithm_data (dict): Stores data necessary for the algorithm. This heuristic does not utilize algorithm_data.
        
        **kwargs: Additional hyperparameters (unused in this heuristic).
        
    Returns:
        AdvanceOperator: The selected operator to advance the job with the shortest next operation time.
        dict: An empty dictionary as this heuristic does not update algorithm_data.
    """
    
    # Extract necessary data from problem_state
    job_operation_time = problem_state["job_operation_time"]
    unfinished_jobs = problem_state["unfinished_jobs"]
    job_operation_index = problem_state["job_operation_index"]
    
    # Initialize variables to store the job with minimum processing time and its time
    min_time = float('inf')
    job_to_advance = None
    
    # Iterate through all unfinished jobs to find the job with the shortest next operation time
    for job_id in unfinished_jobs:
        operation_index = job_operation_index[job_id]
        operation_time = job_operation_time[job_id][operation_index]
        
        # Update the job_to_advance if this job has the shortest next operation time
        if operation_time < min_time:
            min_time = operation_time
            job_to_advance = job_id
    
    # If no job is found (e.g., if there are no unfinished jobs), return None and an empty dict
    if job_to_advance is None:
        return None, {}
    
    # Create and return the AdvanceOperator for the job with the shortest next operation time
    return AdvanceOperator(job_to_advance), {}