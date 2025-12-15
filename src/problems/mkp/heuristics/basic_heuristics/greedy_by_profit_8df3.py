from src.problems.mkp.components import *

def greedy_by_profit_8df3(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AddOperator, dict]:
    """
    Greedy constructive addition by absolute profit under multidimensional feasibility. Items not yet in the knapsack are sorted by profit descending; iteration then uses first-improvement: select the first item in this order whose per-resource weight fits within the current remaining capacities. Feasibility is checked across all resource dimensions via weights[res][item] ≤ remaining_capacity[res], assuming a resource-major weight layout (weights indexed as [resource][item]) and remaining_capacity aligned with capacities.
    Selection policy: first feasible among profit-sorted items; no best-improvement scan and no ratio-based scoring. Returns a single AddOperator for the chosen item.     Scope: single-item addition only; no repair, toggling, or swaps. Complexity: O(k log k + R·k), where k = |items_not_in_knapsack| and R = number of resources; constant extra memory. Tailored to constructive phases where remaining capacities are maintained externally.

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
    # Extract necessary data from problem_state
    profits = problem_state["profits"]
    weights = problem_state["weights"]
    capacities = problem_state["capacities"]
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