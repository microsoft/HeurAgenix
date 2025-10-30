from src.problems.jssp.components import Solution, AdvanceOperator

def shortest_processing_time_first_c374(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """
    Shortest-processing-time (SPT) dispatch for partial JSSP schedules. Each call selects the unfinished job whose immediate next operation has the smallest processing time, then issues an AdvanceOperator to append that operation at the tail of its designated machine’s queue. The rule is job-centric: machine resolution is implicit via the job’s operation sequence; it advances exactly one next operation, preserving job precedence. Tie-breaking follows the first minimum encountered. This is a purely local time-based priority: it ignores machine idleness, queue congestion, and global makespan effects, and performs no reordering on any machine (append-only). Time complexity: O(|unfinished_jobs|); constant memory; algorithm_data unused.
    
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