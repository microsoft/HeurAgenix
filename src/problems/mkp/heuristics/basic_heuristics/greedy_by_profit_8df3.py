from src.problems.mkp.components import *

def greedy_by_profit_8df3(problem_state: dict, algorithm_data: dict, get_state_data_function: callable) -> tuple[AddOperator, dict]:
    """
    Greedy by Profit Heuristic for the Multidimensional Knapsack Problem.
    This heuristic selects items based on the highest profit value until no further items can be added without violating resource constraints.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "profits" (numpy.array): The profit value associated with each item.
            - "weights" (numpy.array): A 2D array where each row represents the resource consumption of an item across all dimensions.
            - "capacities" (numpy.array): The maximum available capacity for each resource dimension.
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "remaining_capacity" (numpy.array): The remaining capacity for each resource dimension after considering the items included in the current solution.            - "items_not_in_knapsack" (list[int]): A list of item indices that are currently not included in the knapsack.

    Returns:
        AddOperator: The operator to add the selected item to the knapsack.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data from global_data
    profits = problem_state["profits"]
    weights = problem_state["weights"]
    capacities = problem_state["capacities"]

    # Extract necessary data from state_data
    current_solution = problem_state["current_solution"]
    remaining_capacity = problem_state["remaining_capacity"]
    items_not_in_knapsack = problem_state["items_not_in_knapsack"]

    # Sort items by profit in descending order, considering only those not in the knapsack
    sorted_items_by_profit = sorted(items_not_in_knapsack, key=lambda i: profits[i], reverse=True)

    for item in sorted_items_by_profit:
        # Check if adding the current item violates any resource constraints
        if all(remaining_capacity[res] >= weights[res][item] for res in range(len(capacities))):
            # If the item can be added without violating constraints, return the corresponding AddOperator
            return AddOperator(item), {}

    # If no items can be added without violating constraints, return None
    return None, {}