from src.problems.mkp.components import *
import random

def random_4c25(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AddOperator, dict]:
    """
    Random Heuristic for the Multidimensional Knapsack Problem.
    Randomly selects items to add to the knapsack until no further items can be added without violating resource constraints.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "weights" (numpy.array): A 2D array where each row represents the resource consumption of an item across all dimensions.
            - "capacities" (numpy.array): The maximum available capacity for each resource dimension.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "remaining_capacity" (numpy.array): The remaining capacity for each resource dimension after considering the items included in the current solution.            - "items_not_in_knapsack" (list[int]): A list of item indices that are currently not included in the knapsack.

    Returns:
        TargetOperatorType: The operator that modifies the current solution towards the heuristic's goal.
        dict: An empty dictionary as this heuristic does not update the algorithm data.
    """
    # Extract necessary data from global_data and state_data
    weights = global_data["weights"]
    capacities = global_data["capacities"]
    remaining_capacity = state_data["remaining_capacity"]
    items_not_in_knapsack = state_data["items_not_in_knapsack"]

    # Shuffle the list of items not in the knapsack to ensure random selection
    random.shuffle(items_not_in_knapsack)

    # Try to add items randomly without violating the resource constraints
    for item in items_not_in_knapsack:
        # Check if the item can be added without exceeding any resource capacity
        if all(remaining_capacity[res] >= weights[res][item] for res in range(len(capacities))):
            # If the item is feasible, create an AddOperator to include the item in the knapsack
            return AddOperator(item), {}

    # If no items can be added without violating constraints, return None
    return None, {}