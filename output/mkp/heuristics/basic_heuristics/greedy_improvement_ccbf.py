from src.problems.mkp.components import *

def greedy_improvement_ccbf(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[SwapOperator, dict]:
    """
    Greedy Improvement heuristic for the Multidimensional Knapsack Problem.
    This heuristic tries to improve the current solution by swapping an included item with an excluded item
    if it results in a higher profit without violating the resource constraints.

    Args:
        global_data (dict): Contains global information about the problem instance.
        state_data (dict): Contains the current state of the solution.
        algorithm_data (dict): Contains data specific to the algorithm's execution.
        get_state_data_function (callable): A function that takes a new solution and returns the corresponding state data.

    Returns:
        SwapOperator: The operator that swaps two items if an improvement is found.
        dict: An empty dictionary as this heuristic does not update algorithm_data.
    """
    # Extract necessary data from global_data
    profits = global_data['profits']
    weights = global_data['weights']
    capacities = global_data['capacities']

    # Extract necessary data from state_data
    current_solution = state_data['current_solution']
    current_weights = state_data['current_weights']
    items_in_knapsack = state_data['items_in_knapsack']
    items_not_in_knapsack = state_data['items_not_in_knapsack']

    # Initialize variables to track the best swap
    best_profit_increase = 0
    best_swap_operator = None

    # Iterate over all pairs of included and excluded items to find the best swap
    for included_item in items_in_knapsack:
        for excluded_item in items_not_in_knapsack:
            # Calculate the profit difference if we were to swap the items
            profit_difference = profits[excluded_item] - profits[included_item]

            # Skip if the swap does not increase the profit
            if profit_difference <= 0:
                continue

            # Check if the swap is feasible by ensuring it does not violate resource constraints
            feasible = True
            for resource_index in range(len(capacities)):
                if current_weights[resource_index] - weights[resource_index][included_item] + weights[resource_index][excluded_item] > capacities[resource_index]:
                    feasible = False
                    break

            # If the swap is feasible and improves the profit, remember it
            if feasible and profit_difference > best_profit_increase:
                best_profit_increase = profit_difference
                best_swap_operator = SwapOperator(included_item, excluded_item)

    # If a beneficial swap was found, return the corresponding operator
    if best_swap_operator:
        return best_swap_operator, {}
    else:
        # No beneficial swap found, return None
        return None, {}