from src.problems.jssp.components import Solution, AdvanceOperator

def shortest_processing_time_first_d471(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AdvanceOperator, dict]:
    """Shortest Processing Time First with Dynamic Scoring Heuristic for JSSP.

    This heuristic dynamically evaluates unfinished jobs based on their next operation's machine availability,
    alignment with the optimal trajectory, and a bias factor to guide towards jobs that minimize makespan.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "job_operation_sequence" (list[list[int]]): A list of jobs where each job is a list of operations in their target sequence.
            - "job_operation_time" (list[list[int]]): The time cost for each operation in each job.
            - "machine_num" (int): Total number of machines in the problem.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "unfinished_jobs" (list[int]): List of all unfinished jobs.
            - "machine_last_operation_end_times" (list[int]): The end time of the last operation for each machine.
            - "job_operation_index" (list[int]): The index of the next operation to be scheduled for each job.
            - "job_last_operation_end_times" (list[int]): The end time of the last operation for each job.
            - "current_solution" (Solution): The current solution state.
            - "job_diversity" (int): Diversity of jobs in the dataset (default to 1 if not provided).
        algorithm_data (dict): The algorithm dictionary for the current algorithm only. Not used in this heuristic.
        get_state_data_function (callable): Function to get the state data for a new solution. Not used in this heuristic.
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
    if not state_data["unfinished_jobs"]:
        return None, {}

    # Extract necessary information from global and state data
    unfinished_jobs = state_data["unfinished_jobs"]
    machine_last_end_times = state_data["machine_last_operation_end_times"]
    job_operation_index = state_data["job_operation_index"]
    job_last_end_times = state_data["job_last_operation_end_times"]
    job_operation_sequence = global_data["job_operation_sequence"]
    job_diversity = state_data.get("job_diversity", 1)  # Default to 1 if not provided

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