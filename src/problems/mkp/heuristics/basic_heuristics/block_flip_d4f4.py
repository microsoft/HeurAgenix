from src.problems.mkp.components import *

def block_flip_d4f4(problem_state: dict, algorithm_data: dict, block_size: int = 2) -> tuple[FlipBlockOperator, dict]:
    """
    Block Flip Heuristic for Multidimensional Knapsack Problem.
    Flips the inclusion status of a contiguous block of items to test for an improved solution.
    
    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weights" (numpy.array): A 2D array where each row represents the resource consumption of an item across all dimensions.
            - "capacities" (numpy.array): The maximum available capacity for each resource dimension.
            - "current_solution" (Solution): The current solution instance.
            - "remaining_capacity" (numpy.array): The remaining capacity for each resource dimension after considering the items included in the current solution.        algorithm_data (dict): Contains the data necessary for the algorithm. Not used in this function.
            - "validation_solution" (callable): validation solution.
        get_state_data_function (callable): Function to get the state data for a new solution.
        block_size (int): The size of the block to consider for flipping. Defaults to 2.

    Returns:
        FlipBlockOperator: The operator to flip a block of items if an improved solution is found.
        dict: Empty dictionary as no algorithm data is updated.
    """

    # Extract necessary data from problem_state
    weights = problem_state['weights']
    capacities = problem_state['capacities']
    current_solution = problem_state['current_solution']
    validation_solution = problem_state['validation_solution']


    # Determine the range for the start of the block
    n_items = len(current_solution.item_inclusion)
    start_range = range(n_items - block_size + 1)

    # Initialize variables to track the best block flip
    best_operator = None
    best_state_data = None
    best_profit = problem_state['current_profit']
    
    # Iterate over all possible starting positions for the block
    for start in start_range:
        # Define the block of items to potentially flip
        block = list(range(start, start + block_size))
        
        # Create a new solution by flipping the selected block
        new_solution = FlipBlockOperator(block).run(current_solution)
        
        # Get the state data for the new solution
        new_state_data = get_state_data_function(new_solution)
        
        # If the new solution is valid and improves the profit, update the best_operator
        if validation_solution(new_solution) and new_problem_state['current_profit'] > best_profit:
            best_profit = new_problem_state['current_profit']
            best_operator = FlipBlockOperator(block)
            best_state_data = new_state_data
    
    # If no improvement is found, return None
    if not best_operator:
        return None, {}
    
    # Return the best operator found and an empty dictionary as no algorithm data is updated
    return best_operator, {}