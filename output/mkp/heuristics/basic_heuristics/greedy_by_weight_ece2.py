from src.problems.mkp.components import *

def greedy_by_weight_ece2(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable) -> tuple[AddOperator, dict]:
    """
    Greedy by Weight heuristic for the Multidimensional Knapsack Problem.
    This heuristic selects items based on the smallest weight for a given dimension,
    adding them to the knapsack until no further items can be added without exceeding
    the capacity for that dimension.

    Args:
        global_data (dict): Contains global information about the problem.
            - "weights" (numpy.array): A 2D array where each row represents the resource consumption of an item across all dimensions.
            - "capacities" (numpy.array): The maximum available capacity for each resource dimension.
        state_data (dict): Contains the current state of the solution.
            - "remaining_capacity" (numpy.array): The remaining capacity for each resource dimension after considering the items included in the current solution.            - "items_not_in_knapsack" (list[int]): Item indices not currently included in the knapsack.
        algorithm_data (dict): Contains data specific to the algorithm, not used in this heuristic.
        get_state_data_function (callable): Function to get the state data for a new solution.

    Returns:
        AddOperator: The operator to add the selected item to the knapsack.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data from global_data and state_data
    weights = global_data["weights"]
    capacities = global_data["capacities"]
    remaining_capacity = state_data["remaining_capacity"]
    items_not_in_knapsack = state_data["items_not_in_knapsack"]

    # Sort items by their weight in the first dimension
    sorted_items_by_weight = sorted(items_not_in_knapsack, key=lambda item: weights[0][item])

    # Iterate over sorted items and try to add the lightest item that fits
    for item in sorted_items_by_weight:
        # Check if the item can be added without violating any resource constraints
        if all(weights[res][item] <= remaining_capacity[res] for res in range(len(capacities))):
            # Create a new solution with the item added
            new_solution = Solution(state_data["current_solution"].item_inclusion[:])
            new_solution.item_inclusion[item] = True
            return AddOperator(item), {}
    
    # If no item can be added, return None
    return None, {}