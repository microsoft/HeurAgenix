from src.problems.online_bpp.components import Solution, AssignBinOperator, NewBinOperator

def first_fit_2a2c(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AssignBinOperator or NewBinOperator, dict]:
    """ FirstFit heuristic algorithm for the Online Bin Packing Problem.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "capacity" (int): The capacity for each bin.
            - "item_num" (int): Total item number.
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "current_item_size" (int): The size of current item to pack.
            - "remaining_capacity" (list[int]): List of remaining capacity for each bin.
        get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.

    Returns:
        An operator (AssignBinOperator or NewBinOperator) that modifies the solution to include the current item in the appropriate bin.
        An empty dictionary as this algorithm does not update algorithm data.
    """
    # Retrieve necessary data from state_data
    current_item_size = problem_state["current_item_size"]
    remaining_capacity = problem_state["remaining_capacity"]

    # If there is no current item to pack, return None
    if current_item_size is None:
        return None, {}

    # Iterate over the list of bins to find the first bin with enough capacity
    for bin_index, capacity in enumerate(remaining_capacity):
        if capacity >= current_item_size:
            # If a suitable bin is found, return an AssignBinOperator for this bin
            return AssignBinOperator(bin=bin_index), {}

    # If no bin can accommodate the current item, create a new bin
    return NewBinOperator(), {}