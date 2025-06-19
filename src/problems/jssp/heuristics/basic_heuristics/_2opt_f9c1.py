from src.problems.jssp.components import Solution, SwapOperator

def _2opt_f9c1(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[SwapOperator, dict]:
    """Implements a 2-opt heuristic for the Job Shop Scheduling Problem (JSSP).
    This heuristic attempts to reduce the makespan by swapping two non-adjacent operations in the schedule.
    It iteratively checks all possible pairs of operations to determine if a shorter sequence can be found.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "job_operation_sequence" (numpy.ndarray): Each job's sequence of operations.
            - "machine_num" (int): The total number of machines.
            - "current_solution" (Solution): The current solution state.
            - "job_operation_index" (list[int]): The index of the next operation to be scheduled for each job.

    Returns:
        (SwapOperator, dict): A tuple where the first element is the selected operator that performs the best swap found,
                              and the second element is an empty dictionary since this algorithm does not update algorithm_data.
    """
    current_solution = problem_state['current_solution']
    job_operation_sequence = problem_state['job_operation_sequence']
    machine_num = problem_state['machine_num']
    best_operator = None
    best_delta = 0

    # Iterating over all machines and all non-adjacent pairs of jobs within each machine's sequence
    for machine_id in range(machine_num):
        for i in range(len(current_solution.job_sequences[machine_id]) - 2):
            for j in range(i + 2, len(current_solution.job_sequences[machine_id])):
                job_id1 = current_solution.job_sequences[machine_id][i]
                job_id2 = current_solution.job_sequences[machine_id][j]

                # Calculate the makespan change if the swap is performed
                new_solution = SwapOperator(machine_id, job_id1, job_id2).run(current_solution)
                new_state = get_state_data_function(new_solution)
                if new_state == None:
                    continue
                delta = new_state['current_makespan'] - problem_state['current_makespan']

                # Check if this is the best swap found so far
                if best_operator is None or delta < best_delta:
                    best_operator = SwapOperator(machine_id, job_id1, job_id2)
                    best_delta = delta

    # If a beneficial swap is found, return the corresponding operator
    if best_operator and best_delta < 0:
        return best_operator, {}
    else:
        return None, {}