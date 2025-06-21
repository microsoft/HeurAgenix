from src.problems.jssp.components import Solution, AdvanceOperator

def first_come_first_served_af26(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """Enhanced First Come First Served (FCFS) heuristic to dynamically prioritize jobs based on immediate impact and adapt to diverse datasets.
    This heuristic introduces a dynamic scoring mechanism to balance alignment with the positive solution trajectory, machine availability, 
    and dataset-specific characteristics.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "job_operation_sequence" (list[list[int]]): A list of jobs where each job is a list of operations in target sequence.
            - "machine_num" (int): Total number of machines in the problem.
            - "unfinished_jobs" (list[int]): List of all unfinished jobs.
            - "machine_last_operation_end_times" (list[int]): The end time of the last operation for each machine.
            - "job_operation_index" (list[int]): The index of the next operation to be scheduled for each job.
            - "job_last_operation_end_times" (list[int]): The end time of the last operation for each job.
            - "job_diversity" (int): Diversity of jobs based on the dataset (optional but improves adaptability).
        algorithm_data (dict): Not used in this heuristic.
        problem_state["get_problem_state"] (callable): Function to get the state data for a new solution (not used in this heuristic).
        kwargs (optional): Hyperparameters for fine-tuning:
            - bias_weight (float, default=50.0): The weight to prioritize jobs aligning with the positive solution trajectory.
            - diversity_threshold (int, default=5): A threshold to determine when to adapt scoring based on job diversity.

    Returns:
        AdvanceOperator: The operator that advances the selected job based on priority.
        dict: Empty dictionary as no algorithm data is updated.
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

    # Determine if fallback to original FCFS logic is necessary based on dataset characteristics
    if job_diversity <= diversity_threshold and len(unfinished_jobs) == 1:
        job_id = unfinished_jobs[0]
        return AdvanceOperator(job_id=job_id), {}

    # Initialize variables for dynamic priority evaluation
    best_job = None
    best_priority_score = float('inf')  # Lower priority score is better

    for job_id in unfinished_jobs:
        # Determine the machine for the next operation of the job
        next_machine_id = job_operation_sequence[job_id][job_operation_index[job_id]]

        # Calculate priority score based on machine availability and job alignment
        priority_score = max(machine_last_end_times[next_machine_id], job_last_end_times[job_id])

        # Introduce a dynamic bias to align with the positive solution trajectory
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