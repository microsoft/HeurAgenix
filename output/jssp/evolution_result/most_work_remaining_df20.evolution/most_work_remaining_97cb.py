from src.problems.jssp.components import Solution, AdvanceOperator, SwapOperator
import numpy as np

def most_work_remaining_97cb(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AdvanceOperator, dict]:
    """Most Work Remaining with Enhanced Optimization for JSSP.

    This heuristic selects jobs based on most remaining work, with enhanced logic for handling high makespan scenarios and optimizing job sequences.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - job_operation_sequence (list[list[int]]): A list of jobs where each job is a list of operations in their target sequence.
            - job_operation_time (list[list[int]]): The time cost for each operation in each job.
            - machine_num (int): Total number of machines in the problem.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - unfinished_jobs (list[int]): List of all unfinished jobs.
            - machine_last_operation_end_times (list[int]): The end time of the last operation for each machine.
            - job_operation_index (list[int]): The index of the next operation to be scheduled for each job.
            - job_last_operation_end_times (list[int]): The end time of the last operation for each job.
            - current_solution (Solution): The current solution state.
            - current_makespan (int): The time cost for current operation jobs.
        algorithm_data (dict): Contains the data necessary for some algorithms, specifically:
            - iteration (int): The current iteration count for the heuristic.
        get_state_data_function (callable): A function to return the state dictionary for a new solution.
        kwargs: Optional hyperparameters including:
            - bias_weight (float, default=60.0): The weight to prioritize jobs aligning with the positive solution trajectory.
            - swap_frequency (int, default=5): Frequency for applying swap optimization.
            - makespan_threshold (int, default=400): Threshold above which swap operations are prioritized.

    Returns:
        An operator that advances the selected job based on priority and optimization logic.
        An updated algorithm data dictionary with the new iteration count.
    """
    # Extract hyperparameters with default values
    bias_weight = kwargs.get("bias_weight", 60.0)
    swap_frequency = kwargs.get("swap_frequency", 5)
    makespan_threshold = kwargs.get("makespan_threshold", 400)

    # Get the current iteration from algorithm_data
    iteration = algorithm_data.get("iteration", 0)

    # Extract necessary data
    unfinished_jobs = state_data["unfinished_jobs"]
    machine_last_end_times = state_data["machine_last_operation_end_times"]
    job_operation_index = state_data["job_operation_index"]
    job_last_end_times = state_data["job_last_operation_end_times"]
    job_operation_sequence = global_data["job_operation_sequence"]
    job_operation_time = global_data["job_operation_time"]
    current_solution = state_data["current_solution"]
    current_makespan = state_data["current_makespan"]
    machine_num = global_data["machine_num"]

    # If no unfinished jobs, return None
    if not unfinished_jobs:
        return None, {}

    # Initialize variables for scoring
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

        # Calculate remaining work for the job
        remaining_work = sum(job_operation_time[job_id][next_operation_index:])

        # Calculate penalty for current makespan
        makespan_penalty = bias_weight if current_makespan > makespan_threshold else 0

        # Calculate total score
        score = expected_machine_time + makespan_penalty - remaining_work
        if score < best_score:
            best_score = score
            best_job = job_id

    # If no best job is found, return None
    if best_job is None:
        return None, {}

    # Apply swap optimization periodically if makespan is above threshold
    if current_makespan > makespan_threshold and iteration % swap_frequency == 0:
        best_operator = None
        best_delta = float("inf")

        # Try swap optimization
        for machine_id in range(machine_num):
            for i in range(len(current_solution.job_sequences[machine_id]) - 1):
                for j in range(i + 1, len(current_solution.job_sequences[machine_id])):
                    job_id1 = current_solution.job_sequences[machine_id][i]
                    job_id2 = current_solution.job_sequences[machine_id][j]
                    new_solution = SwapOperator(machine_id, job_id1, job_id2).run(current_solution)
                    new_state = get_state_data_function(new_solution)
                    if new_state is None:
                        continue
                    delta = new_state["current_makespan"] - state_data["current_makespan"]
                    if delta < best_delta:
                        best_operator = SwapOperator(machine_id, job_id1, job_id2)
                        best_delta = delta

        # If a better operator is found, return it
        if best_operator is not None:
            return best_operator, {"iteration": iteration + 1}

    # Return the best AdvanceOperator
    return AdvanceOperator(job_id=best_job), {"iteration": iteration + 1}