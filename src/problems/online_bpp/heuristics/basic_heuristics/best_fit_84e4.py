from src.problems.online_bpp.components import Solution, AssignBinOperator, NewBinOperator

def best_fit_84e4(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AssignBinOperator | NewBinOperator, dict]:
    """ BestFit heuristic algorithm for the Online Bin Packing Problem.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "capacity" (int): The fixed capacity for each bin.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "current_item_size" (int): The size of the current item to pack.
            - "remaining_capacity" (list[int]): List of remaining capacity for each bin.

    Returns:
        AssignBinOperator or NewBinOperator: The operator that assigns the item to the best-fit bin or a new bin if no existing bin can accommodate the item.
        dict: An empty dictionary as no algorithm-specific data updates are needed.
    """
    current_item_size = state_data["current_item_size"]
    remaining_capacity = state_data["remaining_capacity"]
    capacity = global_data["capacity"]

    # If there is no current item to pack, return None
    if current_item_size is None:
        return None, {}

    # Initialize the index and minimum remaining capacity for the best-fit bin
    best_fit_index = None
    min_remaining_capacity = float('inf')

    # Iterate through bins to find the best-fit bin
    for i, rem_cap in enumerate(remaining_capacity):
        if rem_cap >= current_item_size and rem_cap - current_item_size < min_remaining_capacity:
            best_fit_index = i
            min_remaining_capacity = rem_cap - current_item_size

    # Determine the operator to use
    if best_fit_index is not None:
        # Use the AssignBinOperator if a suitable bin is found
        operator = AssignBinOperator(best_fit_index)
    else:
        # Use the NewBinOperator if no suitable bin is found
        operator = NewBinOperator()

    # Return the operator and an empty dictionary (no algorithm data updates needed)
    return operator, {}