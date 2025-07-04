from src.problems.jssp.components import Solution, AdvanceOperator

def longest_processing_time_first_9dc9(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """Selects the next job to be scheduled based on the Longest Processing Time (LPT) heuristic.
    
    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "job_operation_time" (numpy.ndarray):  The time cost for each operation in each job.
            - "job_num" (int): The total number of jobs.
            - "machine_num" (int): The total number of machines.
            - "unfinished_jobs" (list[int]): List of all unfinished jobs.
            - "current_solution" (Solution): The current solution state.
        
        algorithm_data (dict): Contains data specific to the algorithm (unused in this heuristic).
        
        **kwargs: Additional hyperparameters (unused in this heuristic).
    
    Returns:
        AdvanceOperator: An operator that, when applied to the current solution, advances the longest processing job.
        dict: An empty dictionary since this heuristic doesn't update algorithm_data.
    """
    
    # No additional information from algorithm_data or kwargs is used in this heuristic.
    
    # Extract necessary information from problem_state
    job_operation_time = problem_state["job_operation_time"]
    unfinished_jobs = problem_state["unfinished_jobs"]
    current_solution = problem_state["current_solution"]
    
    # Initialize variables to keep track of the job with the longest next operation time
    max_time = -1
    selected_job = None
    
    # Iterate through all unfinished jobs to find the job with the longest next operation time
    for job_id in unfinished_jobs:
        next_operation_index = current_solution.job_operation_index[job_id]
        
        # Check if there is a next operation for the job
        if next_operation_index < len(job_operation_time[job_id]):
            next_operation_time = job_operation_time[job_id][next_operation_index]
            
            # Update the job with the longest next operation time
            if next_operation_time > max_time:
                max_time = next_operation_time
                selected_job = job_id
    
    # If no job is found (all jobs are finished), return None and an empty dictionary
    if selected_job is None:
        return None, {}
    
    # Create an AdvanceOperator for the job with the longest next operation time
    operator = AdvanceOperator(selected_job)
    
    # Return the operator and an empty dictionary since no algorithm data is updated
    return operator, {}