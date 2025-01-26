from src.problems.mkp.components import *

def greedy_by_cost_benefit_fd45(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AddOperator, dict]:
    """
    Greedy by Cost-Benefit heuristic for the Multidimensional Knapsack Problem.
    This heuristic selects the item that provides the best trade-off between added profit
    and the opportunity cost of reduced remaining capacity for future items.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "profits" (numpy.array): The profit value associated with each item.
            - "weights" (numpy.array): A 2D array where each row represents the resource consumption of an item across all dimensions.
            - "capacities" (numpy.array): The maximum available capacity for each resource dimension.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "remaining_capacity" (numpy.array): The remaining capacity for each resource dimension after considering the items included in the current solution.            - "items_not_in_knapsack" (list[int]): A list of item indices that are currently not included in the knapsack.
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.

    Returns:
        AddOperator: The operator to add the selected item to the knapsack.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data from global_data and state_data
    profits = global_data['profits']
    weights = global_data['weights']
    remaining_capacity = state_data['remaining_capacity']
    items_not_in_knapsack = state_data['items_not_in_knapsack']

    # Initialize variables to track the best item and its cost-benefit value
    best_item_index = None
    best_cost_benefit_value = float('-inf')

    # Iterate over each item not in the knapsack to find the best cost-benefit trade-off
    for item_index in items_not_in_knapsack:
        # Calculate the cost of adding the item, which is the sum of its weights across all dimensions
        item_cost = sum(weights[resource][item_index] for resource in range(len(remaining_capacity)))
        # Calculate the benefit of adding the item, which is its profit
        item_benefit = profits[item_index]
        # Calculate the opportunity cost, which is the reduction in remaining capacity
        opportunity_cost = 0
        for resource in range(len(remaining_capacity)):
            if remaining_capacity[resource] > 0 or (remaining_capacity[resource] == 0 and weights[resource][item_index] == 0):
                opportunity_cost += weights[resource][item_index] / remaining_capacity[resource] if remaining_capacity[resource] > 0 else 0
            else:
                opportunity_cost += float('inf')
        # Calculate the cost-benefit value for the item
        cost_benefit_value = item_benefit - opportunity_cost * item_cost

        # Check if this item has the best cost-benefit value found so far
        if cost_benefit_value > best_cost_benefit_value and opportunity_cost < float('inf'):
            # Check if adding the item would not exceed any of the resource capacities
            if all(remaining_capacity[resource] >= weights[resource][item_index] for resource in range(len(remaining_capacity))):
                best_item_index = item_index
                best_cost_benefit_value = cost_benefit_value

    # If a best item is found, create an AddOperator to add it to the knapsack
    if best_item_index is not None:
        return AddOperator(best_item_index), {}
    else:
        # If no item can be added without exceeding capacities, return None
        return None, {}