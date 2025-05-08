from src.problems.jssp.components import Solution, AdvanceOperator, ShiftOperator
import numpy as np

def shortest_processing_time_first_49ec(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AdvanceOperator, dict]:
    """Shortest Processing Time First with Enhanced Heuristic for JSSP.

    This heuristic dynamically evaluates unfinished jobs based on their next operation's machine availability,
    alignment with the optimal trajectory, and a bias factor to guide towards jobs that minimize makespan.
    It also considers potential shifts in operation positions to optimize the schedule.

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
        algorithm_data (dict): The algorithm dictionary for the current algorithm only. In this algorithm, the following items are necessary:
            - "iteration" (int): The current iteration count for the heuristic.
        get_state_data_function (callable): Function to get the state data for a new solution. Not used in this heuristic.
        kwargs: Optional hyperparameters for fine-tuning:
            - bias_weight (float, default=70.0): The weight to prioritize jobs aligning with the positive solution trajectory.
            - max_no_improve (int, default=5): Number of iterations without improvement before adjusting strategy.

    Returns:
        AdvanceOperator: The operator that advances the selected job based on priority.
        dict: Updated algorithm data with the new iteration count.
    """
    
    # Extract hyperparameters from kwargs with default values
    bias_weight = kwargs.get("bias_weight", 70.0)
    max_no_improve = kwargs.get("max_no_improve", 5)
    
    # Check if there are any unfinished jobs. If not, return None.
    if not state_data["unfinished_jobs"]:
        return None, {}

    # Extract necessary information from global and state data
    unfinished_jobs = state_data["unfinished_jobs"]
    machine_last_end_times = state_data["machine_last_operation_end_times"]
    job_operation_index = state_data["job_operation_index"]
    job_last_end_times = state_data["job_last_operation_end_times"]
    job_operation_sequence = global_data["job_operation_sequence"]
    
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
        priority_score -= bias_weight / (1 + next_operation_index)  # Dynamic adjustment

        # Update the best job based on the computed priority score
        if priority_score < best_priority_score:
            best_priority_score = priority_score
            best_job = job_id

    # If no job is selected, return None
    if best_job is None:
        return None, {}

    # Apply ShiftOperator if no improvement after certain iterations
    iteration = algorithm_data.get("iteration", 0)
    if iteration >= max_no_improve:
        best_operator = None
        best_delta = float("inf")

        for machine_id in range(global_data["machine_num"]):
            for current_position, job_id in enumerate(state_data["current_solution"].job_sequences[machine_id]):
                for new_position in range(len(state_data["current_solution"].job_sequences[machine_id])):
                    if current_position == new_position:
                        continue
                    new_solution = ShiftOperator(machine_id, job_id, new_position).run(state_data["current_solution"])
                    new_state = get_state_data_function(new_solution)
                    if new_state is None:
                        continue
                    delta = new_state["current_makespan"] - state_data["current_makespan"]
                    if delta < best_delta:
                        best_operator = ShiftOperator(machine_id, job_id, new_position)
                        best_delta = delta

        if best_operator is not None and best_delta < 0:
            return best_operator, {"iteration": iteration + 1}

    # Return the best AdvanceOperator
    return AdvanceOperator(job_id=best_job), {"iteration": iteration + 1}