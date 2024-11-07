from src.problems.jssp.components import Solution, SwapOperator

def _3opt_6ee0(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[SwapOperator, dict]:
    """3-opt heuristic for Job Shop Scheduling Problem (JSSP).
    This function does not generate a complete 3-opt operator since JSSP requires operations within a job to be executed in sequence and does not allow reordering of these operations.The heuristic is adapted to generate a series of 2-opt swaps (using the SwapOperator) which approximate a 3-opt move.

    Args:
        global_data (dict): The global data dict containing the global data.
            - "machine_num" (int): The total number of machines.
            - "job_num" (int): The total number of jobs.
            
        state_data (dict): The state dictionary containing the current state information.
            - "current_solution" (Solution): The current solution state.
            - "unfinished_jobs" (list[int]): List of all unfinished jobs.
            - "machine_last_operation_end_times" (list[int]): The end time of the last operation for each machine.
            
        algorithm_data (dict): Not used in this heuristic.
        
        get_state_data_function (callable): Function to get the new state data given a Solution instance.
    
    Returns:
        SwapOperator: The operator to swap two jobs in the schedule.
        dict: Empty dictionary as no algorithm data is updated.
    """
    
    # Check if we have enough jobs to perform a 3-opt move
    if global_data['job_num'] < 3:
        return None, {}  # Not enough jobs to perform a 3-opt
    
    # Extract necessary information from the state data
    current_solution = state_data['current_solution']
    machine_last_operation_end_times = state_data['machine_last_operation_end_times']
    
    # Initialize variables to store the best swap operator and its corresponding makespan improvement
    best_operator = None
    best_improvement = 0
    
    # Iterate over all machines
    for machine_id in range(global_data['machine_num']):
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
                    new_state = get_state_data_function(new_solution)
                    if new_state == None:
                        continue
                    improvement = state_data['current_makespan'] - new_state['current_makespan']
                    
                    # Update the best operator if this swap results in a better makespan
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_operator = SwapOperator(machine_id, job_sequence[i], job_sequence[k])
    
    # Return the best operator found (if any) and an empty dictionary as no algorithm data is updated
    return best_operator, {}