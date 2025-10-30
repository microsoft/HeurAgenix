from src.problems.jssp.components import ShiftOperator

def shift_operator_109f(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[ShiftOperator, dict]:
    """
    Best-improvement intra-machine shift neighborhood for JSSP makespan minimization. For each machine m, for each job j at position p, it evaluates all relocations of j to positions q≠p within m’s queue. Each candidate schedule is materialized via ShiftOperator and passed to get_problem_state; invalid candidates (None) are discarded, so feasibility (including precedence) is enforced externally, not by the operator. The objective is the makespan delta (new − current); the move with the most negative delta over the entire neighborhood is selected. No first-improvement acceptance: the search is exhaustive and returns a move only if it strictly reduces the makespan, otherwise no action. Scope is limited to intra-machine reordering; machine assignments are not altered. Deterministic traversal (implicit tie-handling retains the earliest found with the same delta). Computational cost scales with ∑_m L_m(L_m−1) full schedule evaluations; constant extra memory.
    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "machine_num" (int): The total number of machines.
            - "current_solution" (Solution): The current solution state.
            - "current_makespan" (int): The current makespan of the schedule.
        problem_state["get_problem_state"] (callable): Function to get the new state data given a Solution instance.

    Returns:
        ShiftOperator: An operator that shifts a job in the schedule to achieve a local improvement.
        dict: An empty dictionary as this heuristic does not require algorithm data updates.
    """

    current_solution = problem_state['current_solution']
    machine_num = problem_state['machine_num']
    best_operator = None
    best_delta = float('inf')

    # Iterate over all machines
    for machine_id in range(machine_num):
        # Iterate over all operations in the machine's queue
        for current_position, job_id in enumerate(current_solution.job_sequences[machine_id]):
            # Try shifting the operation to all possible positions
            for new_position in range(len(current_solution.job_sequences[machine_id])):
                # Skip if the position is the same as the current one
                if current_position == new_position:
                    continue
                
                # Create a new solution with the operation shifted to the new position
                new_solution = ShiftOperator(machine_id, job_id, new_position).run(current_solution)
                new_state = problem_state["get_problem_state"](new_solution)
                
                # If the new solution is valid, evaluate its makespan
                if new_state is not None:
                    delta = new_state['current_makespan'] - problem_state['current_makespan']
                    
                    # If the makespan is improved, store this operator
                    if delta < best_delta:
                        best_operator = ShiftOperator(machine_id, job_id, new_position)
                        best_delta = delta

    # If a beneficial shift is found, return the corresponding operator
    if best_operator and best_delta < 0:
        return best_operator, {}
    else:
        return None, {}