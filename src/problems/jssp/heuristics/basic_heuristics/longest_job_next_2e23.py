from src.problems.jssp.components import Solution, AdvanceOperator

def longest_job_next_2e23(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """
    Longest-Remaining-Processing-Time (LRPT) job-level priority rule for JSSP. At each invocation it scans unfinished_jobs and selects the job whose tail sum of operation times (from its current job_operation_index to the end) is maximal. It then issues a single-step AdvanceOperator to append that job’s next operation to its required machine queue, preserving intra-job precedence by advancing only the next operation.
    Distinctive aspects:
    - Uses cumulative remaining time as a global lookahead metric rather than the duration of the next operation, favoring jobs with heavy tails to reduce late-stage makespan risk.
    - Schedules exactly one operation per call; no machine-state or critical-path evaluation is performed, so resource contention is not considered.
    - Tie-breaking is deterministic via job enumeration order as induced by remaining_times construction.
    Complexity and scope:
    - Time complexity per call: O(|unfinished_jobs| × average remaining operations per job) due to tail summations; constant extra memory.
    - Purely constructive step; leaves existing queues intact except for a single append on the target machine.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "job_operation_time" (numpy.ndarray):  The time cost for each operation in each job.
            - "job_num" (int): The total number of jobs in the problem.
            - "unfinished_jobs" (list[int]): List of all unfinished jobs.
            - "job_operation_index" (list[int]): The index of the next operation to be scheduled for each job.
            - "current_solution" (Solution): The current state of the job sequences on each machine.
            
        algorithm_data (dict): Contains data necessary for this algorithm.
            (No specific data needed for this algorithm; can be omitted or passed as an empty dict)
        
        **kwargs: Additional hyperparameters (not used in this algorithm).

    Returns:
        (AdvanceOperator): The operator to advance the job with the longest processing time remaining.
        (dict): Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data from the global and state dictionaries.
    job_operation_time = problem_state["job_operation_time"]
    unfinished_jobs = problem_state["unfinished_jobs"]
    job_operation_index = problem_state["job_operation_index"]
    
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