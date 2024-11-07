from src.problems.mkp.components import *

def greedy_by_resource_balance_52f0(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AddOperator, dict]:
    """
    Greedy by Resource Balance heuristic for the Multidimensional Knapsack Problem.
    This heuristic selects items that help to balance the resource usage across all dimensions.
    It prefers items that use less of the more consumed resources and more of the less consumed ones.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "weights" (numpy.array): A 2D array where each row represents the resource consumption of an item across all dimensions.
            - "capacities" (numpy.array): The maximum available capacity for each resource dimension.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "remaining_capacity" (numpy.array): The remaining capacity for each resource dimension after considering the items included in the current solution.            - "items_not_in_knapsack" (list[int]): A list of item indices that are currently not included in the knapsack.

    Returns:
        AddOperator: The operator to add an item to the knapsack that best balances the resource usage.
        dict: Empty dictionary as no algorithm data is updated.
    """
    weights = global_data['weights']
    capacities = global_data['capacities']
    remaining_capacity = state_data['remaining_capacity']
    items_not_in_knapsack = state_data['items_not_in_knapsack']

    # Calculate the resource utilization ratio for each dimension
    utilization_ratios = [1 - (remaining / capacity) for remaining, capacity in zip(remaining_capacity, capacities)]

    best_item_index = None
    best_balance_score = float('inf')

    # Iterate over each item not in the knapsack
    for item_index in items_not_in_knapsack:
        item_weight = [weights[resource][item_index] for resource in range(len(capacities))]
        # Calculate the balance score for the item
        balance_score = sum([utilization_ratios[resource] * item_weight[resource] for resource in range(len(capacities))])
        # Check if the item can be added without violating constraints
        if all(remaining_capacity[resource] >= item_weight[resource] for resource in range(len(capacities))):
            # Update the best item based on the balance score
            if balance_score < best_balance_score:
                best_balance_score = balance_score
                best_item_index = item_index

    # If a best item is found, return the corresponding AddOperator
    if best_item_index is not None:
        return AddOperator(best_item_index), {}
    else:
        # If no item can be added without violating constraints, return None
        return None, {}