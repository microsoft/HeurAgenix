from src.problems.mkp.components import AddOperator, SwapOperator
import numpy as np

def greedy_by_density_bb0a(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AddOperator, dict]:
    """Greedy by Density with Swapping heuristic for the Multidimensional Knapsack Problem.

    This heuristic first selects items based on their density (profit divided by total weight across all dimensions).
    After attempting to add high-density items, it considers swapping existing items in the knapsack with those outside
    to further improve the solution profitably.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "profits" (numpy.array): The profit value associated with each item.
            - "weights" (numpy.array): A 2D array where each row represents the resource consumption of an item across all dimensions.
            - "capacities" (numpy.array): The maximum available capacity for each resource dimension.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "remaining_capacity" (numpy.array): The remaining capacity for each resource dimension after considering the items included in the current solution.
            - "items_in_knapsack" (list[int]): A list of item indices that are currently included in the knapsack.
            - "items_not_in_knapsack" (list[int]): A list of item indices that are currently not included in the knapsack.
        (Optional and can be omitted if no hyper parameters data) introduction for hyper parameters in kwargs if used:
            - swap_frequency (int, default=1): Frequency at which swap logic is checked.

    Returns:
        AddOperator or SwapOperator: The operator to add an item or swap items to improve the solution.
        dict: Updated algorithm data.
    """
    profits = global_data["profits"]
    weights = global_data["weights"]
    capacities = global_data["capacities"]
    remaining_capacity = state_data["remaining_capacity"]
    items_in_knapsack = state_data["items_in_knapsack"]
    items_not_in_knapsack = state_data["items_not_in_knapsack"]

    # Hyper-parameter for swap frequency
    swap_frequency = kwargs.get('swap_frequency', 1)

    # Swap logic to improve the solution
    best_profit_increase = 0
    best_swap_operator = None

    for included_item in items_in_knapsack:
        for excluded_item in items_not_in_knapsack:
            profit_difference = profits[excluded_item] - profits[included_item]

            if profit_difference <= 0:
                continue

            feasible = True
            for resource_index in range(len(capacities)):
                if (state_data['current_weights'][resource_index] 
                    - weights[resource_index][included_item] 
                    + weights[resource_index][excluded_item] > capacities[resource_index]):
                    feasible = False
                    break

            if feasible and profit_difference > best_profit_increase:
                best_profit_increase = profit_difference
                best_swap_operator = SwapOperator(included_item, excluded_item)

    # Calculate densities for items not in the knapsack
    densities = []
    for item_index in items_not_in_knapsack:
        item_profit = profits[item_index]
        item_weight_sum = sum(weights[resource_index][item_index] for resource_index in range(global_data["resource_num"]))
        item_density = item_profit / item_weight_sum if item_weight_sum > 0 else 0
        densities.append((item_density, item_index))

    # Sort items by density in descending order
    densities.sort(reverse=True, key=lambda x: x[0])

    # Try to add the item with the highest density without violating the constraints
    for density, item_index in densities:
        if all(weights[resource_index][item_index] <= remaining_capacity[resource_index] for resource_index in range(global_data["resource_num"])):
            return AddOperator(item_index), {}

    if best_swap_operator:
        return best_swap_operator, {}

    # If no items can be added or swapped to improve the solution, return None
    return None, {}