from src.problems.online_bpp.components import Solution, AssignBinOperator, NewBinOperator

def worst_fit_325f(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AssignBinOperator, dict]:
    """ Heuristic algorithm for the Worst Fit strategy in the online bin packing problem.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "capacity" (int): The capacity for each bin.
            - "item_num" (int): Total item number.
        
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "current_item_size" (int): The size of the current item to pack.
            - "remaining_capacity" (list[int]): List of remaining capacity for each bin.

    Returns:
        AssignBinOperator or NewBinOperator based on the Worst Fit strategy.
        An empty dictionary as no algorithm data needs to be updated.
    """
    # Extract necessary information from state_data
    current_item_size = problem_state["current_item_size"]
    remaining_capacity = problem_state["remaining_capacity"]
    current_solution = problem_state["current_solution"]

    # If there is no current item to pack, return None
    if current_item_size is None:
        return None, {}

    # Initialize variables to track the best bin and maximum remaining capacity
    best_bin = -1
    max_remaining_capacity = -1
    
    # Iterate over each bin to find the one with the most remaining capacity after placing the item
    for bin_index, capacity in enumerate(remaining_capacity):
        if capacity >= current_item_size:  # Check if the item can fit in the bin
            if capacity - current_item_size > max_remaining_capacity:
                best_bin = bin_index
                max_remaining_capacity = capacity - current_item_size
    
    # If a suitable bin is found, use AssignBinOperator to place the item
    if best_bin != -1:
        operator = AssignBinOperator(best_bin)
    else:
        # If no suitable bin is found, use NewBinOperator to open a new bin
        operator = NewBinOperator()

    return operator, {}