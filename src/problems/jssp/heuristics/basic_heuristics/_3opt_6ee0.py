from src.problems.jssp.components import Solution, SwapOperator

def _3opt_6ee0(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[SwapOperator, dict]:
    """
    Machine-local 3-opt approximation via non-adjacent 2-opt swap. For each machine with at least three scheduled jobs, iterate all triplets of positions i < j < k and evaluate swapping the jobs at i and k (j serves only to enforce non-adjacency, mimicking a 3-edge reconnection without violating job precedence). Only machine-local swaps are considered; job operation order across machines remains intact.
    Selection policy: best-improvement. Among all evaluated (machine_id, i, k) pairs across all machines, choose the swap that yields the largest makespan reduction. Validity is enforced by recomputing the problem state via get_problem_state; invalid schedules are discarded.
    Required problem_state items: "machine_num", "job_num" (used for an early size check), "current_solution", "get_problem_state", "current_makespan". No algorithm_data is used.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "machine_num" (int): The total number of machines.
            - "job_num" (int): The total number of jobs.
            - "current_solution" (Solution): The current solution state.
            - "unfinished_jobs" (list[int]): List of all unfinished jobs.
            - "machine_last_operation_end_times" (list[int]): The end time of the last operation for each machine.
            - get_problem_state (callable): def validation_solution(solution: Solution) -> bool: The function to get the problem state for given solution without modify it.
            
        algorithm_data (dict): Not used in this heuristic.
    
    Returns:
        SwapOperator: The operator to swap two jobs in the schedule.
        dict: Empty dictionary as no algorithm data is updated.
    """
    
    # Check if we have enough jobs to perform a 3-opt move
    if problem_state['job_num'] < 3:
        return None, {}  # Not enough jobs to perform a 3-opt
    
    # Extract necessary information from the problem state
    current_solution = problem_state['current_solution']
    machine_last_operation_end_times = problem_state['machine_last_operation_end_times']
    
    # Initialize variables to store the best swap operator and its corresponding makespan improvement
    best_operator = None
    best_improvement = 0
    
    # Iterate over all machines
    for machine_id in range(problem_state['machine_num']):
        job_sequence = current_solution.job_sequences[machine_id]
        num_jobs = len(job_sequence)
        
        # Need at least three jobs on the machine to consider 3-opt
        if num_jobs < 3:
            continue
        
        # Consider all triplets of jobs for potential 3-opt moves
        for i in range(num_jobs - 2):
            for j in range(i + 1, num_jobs - 1):
                for k in range(j + 1, num_jobs):
                    # Perform 2-opt swaps and check for improvement
                    new_solution = SwapOperator(machine_id, job_sequence[i], job_sequence[k]).run(current_solution)
                    new_problem_state = problem_state["get_problem_state"](new_solution)
                    if new_problem_state == None:
                        continue
                    improvement = problem_state['current_makespan'] - new_problem_state['current_makespan']
                    
                    # Update the best operator if this swap results in a better makespan
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_operator = SwapOperator(machine_id, job_sequence[i], job_sequence[k])
    
    # Return the best operator found (if any) and an empty dictionary as no algorithm data is updated
    return best_operator, {}