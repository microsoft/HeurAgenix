from src.problems.mkp.components import *

def greedy_by_resource_balance_52f0(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AddOperator, dict]:
    """
    Resource-balance greedy add. Computes per-dimension utilization u_d = used_d / capacity_d = 1 − remaining_d / capacity_d, then scores each candidate item i by s_i = Σ_d u_d · w_{d,i}. Among feasible items (respecting remaining capacity), select the item with the minimum score, i.e., the one least increasing load on currently stressed resources and most leveraging slack dimensions. Selection policy: best-improvement across all feasible candidates; no profit is considered, making this suitable for repair/diversification to mitigate bottleneck dimensions.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weights" (numpy.array): A 2D array where each row represents the resource consumption of an item across all dimensions.
            - "capacities" (numpy.array): The maximum available capacity for each resource dimension.
            - "remaining_capacity" (numpy.array): The remaining capacity for each resource dimension after considering the items included in the current solution.            - "items_not_in_knapsack" (list[int]): A list of item indices that are currently not included in the knapsack.

    Returns:
        AddOperator: The operator to add an item to the knapsack that best balances the resource usage.
        dict: Empty dictionary as no algorithm data is updated.
    """
    weights = problem_state['weights']
    capacities = problem_state['capacities']
    remaining_capacity = problem_state['remaining_capacity']
    items_not_in_knapsack = problem_state['items_not_in_knapsack']

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