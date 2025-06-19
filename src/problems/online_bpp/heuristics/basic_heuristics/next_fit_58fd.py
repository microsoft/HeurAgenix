from src.problems.online_bpp.components import Solution, AssignBinOperator, NewBinOperator

def next_fit_58fd(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[NewBinOperator, dict]:
    """ NextFit heuristic algorithm for the Online Bin Packing Problem.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - capacity (int): The capacity for each bin.
            - current_solution (Solution): An instance of the Solution class representing the current solution.
            - current_item_size (int): The size of the current item to pack.
            - used_bin_num (int): The number of bins that have been used.
            - used_capacity (list[int]): List of used capacities for each bin.
        (Optional and can be omitted if no used) get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution, and it will not modify the original solution.

    Returns:
        The operator used to place the current item in the appropriate bin, either using the existing bin or creating a new one if necessary.
        An empty dictionary, as this heuristic does not update algorithm data.
    """
    # Extract necessary data from the input dictionaries
    capacity = problem_state["capacity"]
    current_item_size = problem_state["current_item_size"]
    used_capacity = problem_state["used_capacity"]
    used_bin_num = problem_state["used_bin_num"]

    # If there is no current item to pack, return None
    if current_item_size is None:
        return None, {}

    # Check if the current item can fit in the current bin
    if used_bin_num > 0 and current_item_size <= capacity - used_capacity[used_bin_num - 1]:
        # If it fits, use the AssignBinOperator for the current bin
        operator = AssignBinOperator(bin=used_bin_num - 1)
    else:
        # If it doesn't fit, use the NewBinOperator to open a new bin
        operator = NewBinOperator()

    # Return the chosen operator and an empty dictionary for algorithm data updates
    return operator, {}