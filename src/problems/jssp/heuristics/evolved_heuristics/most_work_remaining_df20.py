from src.problems.jssp.components import Solution, AdvanceOperator, SwapOperator, ShiftOperator

def most_work_remaining_df20(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """Most Work Remaining Heuristic with Dynamic Scoring and Local Optimization for JSSP.

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
            - "get_problem_state" (callable): def validation_solution(solution: Solution) -> bool: The function to get the problem state for given solution without modify it.
        algorithm_data (dict): The algorithm dictionary for the current algorithm only. In this algorithm, the following items are necessary:
            - "iteration" (int): The current iteration count for the heuristic.
        kwargs (optional): Hyperparameters for fine-tuning:
            - bias_weight (float, default=50.0): The weight to prioritize jobs aligning with the positive solution trajectory.
            - k_flip_frequency (int, default=10): Frequency (iterations) for applying k-flip optimization.
            - swap_frequency (int, default=10): Frequency (iterations) for applying swap optimization.

    Returns:
        AdvanceOperator: The operator that advances the selected job based on priority.
        dict: Updated algorithm data with the new iteration count.
    """
    # Extract hyperparameters with default values
    bias_weight = kwargs.get("bias_weight", 40.0)
    k_flip_frequency = kwargs.get("k_flip_frequency", 10)
    swap_frequency = kwargs.get("swap_frequency", 10)

    # Get the current iteration from algorithm_data
    iteration = algorithm_data.get("iteration", 0)

    # Extract necessary data
    unfinished_jobs = problem_state["unfinished_jobs"]
    machine_last_end_times = problem_state["machine_last_operation_end_times"]
    job_operation_index = problem_state["job_operation_index"]
    job_last_end_times = problem_state["job_last_operation_end_times"]
    job_operation_sequence = problem_state["job_operation_sequence"]
    current_solution = problem_state["current_solution"]
    machine_num = problem_state["machine_num"]

    # If no unfinished jobs, return None
    if not unfinished_jobs:
        return None, {}

    # Initialize variables for dynamic scoring
    best_job = None
    best_score = float("inf")  # Lower score is better

    # Evaluate each unfinished job
    for job_id in unfinished_jobs:
        # Get the machine for the next operation of this job
        next_operation_index = job_operation_index[job_id]
        if next_operation_index >= len(job_operation_sequence[job_id]):
            continue  # Skip jobs with no remaining operations
        next_machine_id = job_operation_sequence[job_id][next_operation_index]

        # Calculate expected machine availability
        expected_machine_time = max(machine_last_end_times[next_machine_id], job_last_end_times[job_id])

        # Add alignment penalty to prioritize jobs that align better with the optimal trajectory
        alignment_penalty = bias_weight / (1 + next_operation_index)

        # Calculate total score
        score = expected_machine_time + alignment_penalty
        if score < best_score:
            best_score = score
            best_job = job_id

    # If no best job is found, return None
    if best_job is None:
        return None, {}

    # Apply local optimization periodically
    if iteration % k_flip_frequency == 0 or iteration % swap_frequency == 0:
        best_operator = None
        best_delta = float("inf")

        # Try k-flip optimization
        for machine_id in range(machine_num):
            for i in range(len(current_solution.job_sequences[machine_id]) - 2):
                for j in range(i + 2, len(current_solution.job_sequences[machine_id])):
                    job_id1 = current_solution.job_sequences[machine_id][i]
                    job_id2 = current_solution.job_sequences[machine_id][j]
                    new_solution = SwapOperator(machine_id, job_id1, job_id2).run(current_solution)
                    new_state = problem_state["get_problem_state"](new_solution)
                    if new_state is None:
                        continue
                    delta = new_state["current_makespan"] - problem_state["current_makespan"]
                    if delta < best_delta:
                        best_operator = SwapOperator(machine_id, job_id1, job_id2)
                        best_delta = delta

        # Try swap optimization
        for machine_id in range(machine_num):
            for current_position, job_id in enumerate(current_solution.job_sequences[machine_id]):
                for new_position in range(len(current_solution.job_sequences[machine_id])):
                    if current_position == new_position:
                        continue
                    new_solution = ShiftOperator(machine_id, job_id, new_position).run(current_solution)
                    new_state = problem_state["get_problem_state"](new_solution)
                    if new_state is None:
                        continue
                    delta = new_state["current_makespan"] - problem_state["current_makespan"]
                    if delta < best_delta:
                        best_operator = ShiftOperator(machine_id, job_id, new_position)
                        best_delta = delta

        # If a better operator is found, return it
        if best_operator is not None:
            return best_operator, {"iteration": iteration + 1}

    # Return the best AdvanceOperator
    return AdvanceOperator(job_id=best_job), {"iteration": iteration + 1}