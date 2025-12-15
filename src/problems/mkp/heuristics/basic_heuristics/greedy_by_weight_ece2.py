from src.problems.mkp.components import *

def greedy_by_weight_ece2(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AddOperator, dict]:
    """
    Greedy first-fit by a single resource dimension. Items not yet in the knapsack are sorted ascending by weight on resource 0 (weights indexed as weights[resource][item]). The heuristic scans this order and selects the first item that is feasible across all resources with respect to remaining_capacity. Selection policy: first-improvement (first feasible after sorting), not best-improvement. Profit is ignored; ranking is driven solely by the chosen resource dimension, with other dimensions used only for feasibility filtering.
    Required problem_state items: "weights" (2D array indexed by [resource][item]), "capacities" (used for resource count), "remaining_capacity" (per-resource available capacity vector aligned to capacities), "items_not_in_knapsack", "current_solution". No algorithm_data is used.
    Neighborhood and operator: returns a single AddOperator for the selected item; constructive step toward filling capacity.


    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weights" (numpy.array): A 2D array where each row represents the resource consumption of an item across all dimensions.
            - "capacities" (numpy.array): The maximum available capacity for each resource dimension.
            - "remaining_capacity" (numpy.array): The remaining capacity for each resource dimension after considering the items included in the current solution.            - "items_not_in_knapsack" (list[int]): Item indices not currently included in the knapsack.
        algorithm_data (dict): Contains data specific to the algorithm, not used in this heuristic.
        problem_state["get_problem_state"] (callable): Function to get the state data for a new solution.

    Returns:
        AddOperator: The operator to add the selected item to the knapsack.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data from problem_state
    weights = problem_state["weights"]
    capacities = problem_state["capacities"]
    remaining_capacity = problem_state["remaining_capacity"]
    items_not_in_knapsack = problem_state["items_not_in_knapsack"]

    # Sort items by their weight in the first dimension
    sorted_items_by_weight = sorted(items_not_in_knapsack, key=lambda item: weights[0][item])

    # Iterate over sorted items and try to add the lightest item that fits
    for item in sorted_items_by_weight:
        # Check if the item can be added without violating any resource constraints
        if all(weights[res][item] <= remaining_capacity[res] for res in range(len(capacities))):
            # Create a new solution with the item added
            new_solution = Solution(problem_state["current_solution"].item_inclusion[:])
            new_solution.item_inclusion[item] = True
            return AddOperator(item), {}
    
    # If no item can be added, return None
    return None, {}