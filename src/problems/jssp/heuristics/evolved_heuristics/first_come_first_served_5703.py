from src.problems.jssp.components import Solution, AdvanceOperator

def first_come_first_served_5703(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AdvanceOperator, dict]:
    """Enhanced First Come First Served heuristic with dynamic scoring for JSSP.
    
    This heuristic prioritizes jobs based on their alignment with the optimal trajectory, machine availability, and specific dataset characteristics.
    It introduces penalties and secondary conditions to handle cases where multiple jobs have similar priority scores.

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
            - "current_makespan" (int): The time cost for current operation jobs, also known as the current makespan.
        kwargs: Optional hyperparameters for fine-tuning:
            - bias_weight (float, default=60.0): The weight to prioritize jobs aligning with the positive solution trajectory.
            - penalty_threshold (int, default=120): Threshold for considering a machine heavily loaded.
            - makespan_threshold (int, default=100): Threshold for applying penalty logic.

    Returns:
        AdvanceOperator: The operator that advances the selected job based on priority.
        dict: Updated algorithm data (empty in this case).
    """

    # Extract hyperparameters with default values
    bias_weight = kwargs.get("bias_weight", 60.0)
    penalty_threshold = kwargs.get("penalty_threshold", 120)
    makespan_threshold = kwargs.get("makespan_threshold", 100)

    # Check if there are any unfinished jobs
    if not state_data["unfinished_jobs"]:
        return None, {}

    # Extract necessary information from global and state data
    unfinished_jobs = state_data["unfinished_jobs"]
    machine_last_end_times = state_data["machine_last_operation_end_times"]
    job_operation_index = state_data["job_operation_index"]
    job_last_end_times = state_data["job_last_operation_end_times"]
    job_operation_sequence = global_data["job_operation_sequence"]
    job_operation_time = global_data["job_operation_time"]

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

        # Introduce a penalty for jobs with higher operation index on heavily loaded machines
        if state_data["current_makespan"] > makespan_threshold and machine_last_end_times[next_machine_id] > penalty_threshold:
            priority_score += job_operation_index[job_id] * 10  # Arbitrary penalty factor

        # Introduce a dynamic bias to align with the positive solution trajectory
        priority_score -= bias_weight / (1 + job_operation_index[job_id])  # Dynamic adjustment based on operation index

        # Secondary condition to break ties: least total processing time remaining
        remaining_time = sum(job_operation_time[job_id][next_operation_index:])
        if priority_score < best_priority_score or (priority_score == best_priority_score and remaining_time < best_priority_score):
            best_priority_score = priority_score
            best_job = job_id

    # If no job is selected, return None
    if best_job is None:
        return None, {}

    # Create and return the AdvanceOperator for the selected job
    operator = AdvanceOperator(job_id=best_job)
    return operator, {}