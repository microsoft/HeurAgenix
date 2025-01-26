from src.problems.mkp.components import *
from itertools import combinations

def greedy_by_density_4c11(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AddOperator, dict]:
    """Greedy by Density with Enhanced Metrics for the Multidimensional Knapsack Problem (MKP).

    This heuristic selects items based on a weighted density metric and incorporates feasibility checks, lookahead mechanisms, and periodic flip/swap optimizations to improve decision-making.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "profits" (numpy.array): The profit value associated with each item.
            - "weights" (numpy.array): A 2D array where each row represents the resource consumption of an item across all dimensions.
            - "capacities" (numpy.array): The maximum available capacity for each resource dimension.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "remaining_capacity" (numpy.array): The remaining capacity for each resource dimension after considering the items included in the current solution.
            - "items_not_in_knapsack" (list[int]): A list of item indices that are currently not included in the knapsack.
            - "current_solution" (Solution): The current solution instance.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary:
            - "step_count" (int): The current step count of the algorithm.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution, ensuring the original solution remains unmodified.
        kwargs:
            - beta (float): Weighting factor for profit in the density metric. Default is 0.1.
            - gamma (float): Penalty factor for items blocking high-profit items. Default is 0.5.
            - flip_frequency (int): Frequency of performing flip operations. Default is 5.
            - k_flip (int): Number of items to flip during flip operations. Default is 2.

    Returns:
        AddOperator: The operator to add the selected item to the knapsack.
        dict: Updated algorithm data with the incremented step count.
    """

    # Hyper-parameter defaults
    beta = kwargs.get("beta", 0.1)
    gamma = kwargs.get("gamma", 0.5)
    flip_frequency = kwargs.get("flip_frequency", 5)
    k_flip = kwargs.get("k_flip", 2)

    # Extract necessary data
    profits = global_data["profits"]
    weights = global_data["weights"]
    capacities = global_data["capacities"]
    remaining_capacity = state_data["remaining_capacity"]
    items_not_in_knapsack = state_data["items_not_in_knapsack"]
    current_solution = state_data["current_solution"]
    step_count = algorithm_data.get("step_count", 0)

    # Initialize variables
    densities = []
    best_operator = None
    best_profit = float("-inf")

    # Calculate weighted density for each item not in the knapsack
    for item_index in items_not_in_knapsack:
        item_profit = profits[item_index]
        item_weight_sum = sum(weights[resource_index][item_index] for resource_index in range(len(capacities)))
        item_density = (item_profit / item_weight_sum) + beta * (item_profit / profits.sum())
        densities.append((item_density, item_index))

    # Sort items by adjusted density
    densities.sort(reverse=True, key=lambda x: x[0])

    # Feasibility and lookahead checks
    for density, item_index in densities:
        # Simulate adding the item and check its feasibility
        simulated_remaining_capacity = remaining_capacity - weights[:, item_index]
        if all(simulated_remaining_capacity >= 0):
            # Check if adding this item blocks high-profit items
            high_profit_items = [i for i in items_not_in_knapsack if profits[i] > profits[item_index]]
            if any(all(weights[resource_index][hp_item] <= simulated_remaining_capacity[resource_index] for resource_index in range(len(capacities))) for hp_item in high_profit_items):
                continue  # Skip adding this item if it blocks high-profit items

            # If feasible and beneficial, select this item
            return AddOperator(item_index), {"step_count": step_count + 1}

    # Periodic k-flip optimization
    if step_count % flip_frequency == 0 and len(items_not_in_knapsack) >= k_flip:
        for indices_to_flip in combinations(items_not_in_knapsack, k_flip):
            new_solution = current_solution.item_inclusion[:]
            for index in indices_to_flip:
                new_solution[index] = not new_solution[index]
            new_state_data = get_state_data_function(Solution(new_solution))
            if new_state_data and new_state_data["current_profit"] > state_data["current_profit"]:
                return FlipBlockOperator(list(indices_to_flip)), {"step_count": step_count + 1}

    # Swap-based lookahead
    for item_in in state_data.get("items_in_knapsack", []):
        for item_out in items_not_in_knapsack:
            new_solution = current_solution.item_inclusion[:]
            new_solution[item_in], new_solution[item_out] = new_solution[item_out], new_solution[item_in]
            new_state_data = get_state_data_function(Solution(new_solution))
            if new_state_data and new_state_data["current_profit"] > state_data["current_profit"]:
                return SwapOperator(item_in, item_out), {"step_count": step_count + 1}

    # If no valid operator found, return None
    return None, {"step_count": step_count + 1}