from src.problems.mkp.components import AddOperator
import numpy as np

def greedy_by_weight_566d(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, profit_variance_threshold: float = 1000000.0, ratio_adjustment_factor: float = 0.5, capacity_adjustment_factor: float = 0.3, **kwargs) -> tuple[AddOperator, dict]:
    """ Greedy heuristic for the Multidimensional Knapsack Problem, prioritizing items that maximize profit given the remaining capacities.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - profits (numpy.array): A list of profit values for each item.
            - weights (list[list[float]]): A 2D list where each sublist represents the resource consumption of an item across all dimensions.
            - resource_num (int): The total number of resource dimensions.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - remaining_capacity (list[float]): A list of remaining capacities for each resource dimension.
            - items_not_in_knapsack (list[int]): List of indices of items not currently in the knapsack.
        (Optional and can be omitted if no used) get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution, and it will not modify the original solution.
        profit_variance_threshold (float, optional): Threshold for profit variance to prioritize high-profit items. Default is 1000000.0.
        ratio_adjustment_factor (float, optional): Adjustment factor for profit-to-weight ratio based on profit variance. Default is 0.5.
        capacity_adjustment_factor (float, optional): Adjustment factor for the impact of remaining capacity on item selection. Default is 0.3.

    Returns:
        AddOperator: The operator to add the selected item to the knapsack.
        dict: Updated algorithm data (empty in this case).
    """
    # Extract necessary data
    profits = global_data["profits"]
    weights = global_data["weights"]
    resource_num = global_data["resource_num"]
    remaining_capacity = state_data["remaining_capacity"]
    items_not_in_knapsack = state_data["items_not_in_knapsack"]

    # Calculate profit variance
    profit_variance = np.var(profits)

    # Adjusted profit-to-weight ratio considering remaining capacity
    def adjusted_ratio(item):
        weight_sum = sum(weights[res][item] for res in range(resource_num))
        capacity_factor = min((remaining_capacity[res] - weights[res][item]) / remaining_capacity[res] for res in range(resource_num) if remaining_capacity[res] > 0)
        return (profits[item] / weight_sum) * (1 + ratio_adjustment_factor * (profit_variance / profit_variance_threshold)) * (1 + capacity_adjustment_factor * capacity_factor)

    # Sort items by adjusted ratio
    sorted_items_by_adjusted_ratio = sorted(
        items_not_in_knapsack, 
        key=adjusted_ratio,
        reverse=True
    )

    # Try to add the best item by adjusted ratio without violating constraints
    for item in sorted_items_by_adjusted_ratio:
        if all(weights[res][item] <= remaining_capacity[res] for res in range(resource_num)):
            return AddOperator(item), {}

    # If no valid operator was found, return None
    return None, {}