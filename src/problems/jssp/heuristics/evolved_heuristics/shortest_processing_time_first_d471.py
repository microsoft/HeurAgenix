from src.problems.jssp.components import Solution, AdvanceOperator

def shortest_processing_time_first_d471(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """
    Earliest-ready-time dispatch with uniform diversity offset and simple fallback. At each step it scans all unfinished jobs and selects the job whose next operation can start earliest, using priority_score = max(machine_last_end_times[next_machine], job_last_end_times[job]); selection is best-improvement (minimum priority_score) over all candidates. A global bias term −bias_weight/(job_diversity+1) is applied uniformly to all jobs; it shifts scores but does not change the ordering. When job_diversity ≤ diversity_threshold and there is exactly one unfinished job, it immediately advances that job. The AdvanceOperator appends the chosen job to its next machine’s queue and increments the job’s operation index (incremental, open schedule construction). Processing times are not used directly in ranking; readiness is inferred from accumulated end times. Time complexity per decision: O(|unfinished_jobs|); constant extra memory; no algorithm_data updates.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "job_operation_sequence" (list[list[int]]): A list of jobs where each job is a list of operations in their target sequence.
            - "job_operation_time" (list[list[int]]): The time cost for each operation in each job.
            - "machine_num" (int): Total number of machines in the problem.
            - "unfinished_jobs" (list[int]): List of all unfinished jobs.
            - "machine_last_operation_end_times" (list[int]): The end time of the last operation for each machine.
            - "job_operation_index" (list[int]): The index of the next operation to be scheduled for each job.
            - "job_last_operation_end_times" (list[int]): The end time of the last operation for each job.
            - "current_solution" (Solution): The current solution state.
            - "job_diversity" (int): Diversity of jobs in the dataset (default to 1 if not provided).
        algorithm_data (dict): The algorithm dictionary for the current algorithm only. Not used in this heuristic.
        kwargs: Optional hyperparameters for fine-tuning:
            - bias_weight (float, default=50.0): The weight to prioritize jobs aligning with the positive solution trajectory.
            - diversity_threshold (int, default=5): A threshold to determine when to adapt scoring based on job diversity.

    Returns:
        AdvanceOperator: The operator that advances the selected job based on priority.
        dict: An empty dictionary as no algorithm data is updated.
    """
    
    # Extract hyperparameters from kwargs with default values
    bias_weight = kwargs.get("bias_weight", 50.0)
    diversity_threshold = kwargs.get("diversity_threshold", 5)

    # Check if there are any unfinished jobs. If not, return None.
    if not problem_state["unfinished_jobs"]:
        return None, {}

    # Extract necessary information from global and state data
    unfinished_jobs = problem_state["unfinished_jobs"]
    machine_last_end_times = problem_state["machine_last_operation_end_times"]
    job_operation_index = problem_state["job_operation_index"]
    job_last_end_times = problem_state["job_last_operation_end_times"]
    job_operation_sequence = problem_state["job_operation_sequence"]
    job_diversity = problem_state.get("job_diversity", 1)  # Default to 1 if not provided

    # Determine if fallback to a simpler logic is necessary based on dataset characteristics
    if job_diversity <= diversity_threshold and len(unfinished_jobs) == 1:
        job_id = unfinished_jobs[0]
        return AdvanceOperator(job_id=job_id), {}

    # Initialize variables for dynamic priority evaluation
    best_job = None
    best_priority_score = float('inf')  # Lower priority score is better

    for job_id in unfinished_jobs:
        # Determine the machine for the next operation of the job
        next_operation_index = job_operation_index[job_id]
        if next_operation_index >= len(job_operation_sequence[job_id]):
            continue  # Skip jobs that have no remaining operations
        next_machine_id = job_operation_sequence[job_id][next_operation_index]

        # Calculate priority score based on machine availability and job alignment
        priority_score = max(machine_last_end_times[next_machine_id], job_last_end_times[job_id])

        # Introduce a dynamic bias to prefer jobs that align with the positive solution trajectory
        priority_score -= bias_weight / (job_diversity + 1)  # Dynamic adjustment based on diversity

        # Update the best job based on the computed priority score
        if priority_score < best_priority_score:
            best_priority_score = priority_score
            best_job = job_id

    # If no job is selected, return None
    if best_job is None:
        return None, {}

    # Create and return the AdvanceOperator for the selected job
    operator = AdvanceOperator(job_id=best_job)
    return operator, {}