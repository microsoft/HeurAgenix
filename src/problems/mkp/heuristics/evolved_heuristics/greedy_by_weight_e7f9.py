from src.problems.mkp.components import *
from itertools import combinations

def greedy_by_weight_e7f9(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, k_flip_range=(2, 3), **kwargs) -> tuple[AddOperator, dict]:
    """Greedy heuristic with profit-to-weight ratio, k-flip, and swap refinement for MKP.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "profits" (list[float]): A list of profit values for each item.
            - "weights" (list[list[float]]): A 2D list where each sublist represents the resource consumption of an item across all dimensions.
            - "resource_num" (int): The total number of resource dimensions.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current solution object.
            - "current_profit" (float): The total profit of the current solution.
            - "remaining_capacity" (list[float]): A list of remaining capacities for each resource dimension.
            - "items_in_knapsack" (list[int]): List of indices of items currently in the knapsack.
            - "items_not_in_knapsack" (list[int]): List of indices of items not currently in the knapsack.
            - "feasible_items_to_add" (list[int]): List of indices of items that can be added without violating constraints.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. This is not updated in this heuristic.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution.
        k_flip_range (tuple, optional): The range of k values to explore for k-flip. Default is (2, 3).

    Returns:
        AddOperator or FlipBlockOperator or SwapOperator: The operator to apply the improvement.
        dict: Updated algorithm data (empty in this case).
    """
    # Extract necessary data
    profits = global_data["profits"]
    weights = global_data["weights"]
    resource_num = global_data["resource_num"]
    current_solution = state_data["current_solution"]
    remaining_capacity = state_data["remaining_capacity"]
    items_in_knapsack = state_data["items_in_knapsack"]
    items_not_in_knapsack = state_data["items_not_in_knapsack"]
    feasible_items_to_add = state_data["feasible_items_to_add"]
    current_profit = state_data["current_profit"]

    # Step 1: Sort items by profit-to-weight ratio
    if not feasible_items_to_add:
        return None, {}
    sorted_items_by_ratio = sorted(
        feasible_items_to_add, 
        key=lambda item: profits[item] / sum(weights[res][item] for res in range(resource_num)),
        reverse=True
    )

    # Step 2: Try to add the best item by ratio
    for item in sorted_items_by_ratio:
        if all(weights[res][item] <= remaining_capacity[res] for res in range(resource_num)):
            return AddOperator(item), {}

    # Step 3: Attempt k-flip to improve the solution
    best_operator = None
    best_profit = current_profit
    for k in range(k_flip_range[0], k_flip_range[1] + 1):  # Explore k values in the provided range
        for indices_to_flip in combinations(feasible_items_to_add, k):
            # Generate a new solution by flipping the k items
            new_solution = current_solution.item_inclusion[:]
            for index in indices_to_flip:
                new_solution[index] = not new_solution[index]

            # Check if the new solution is valid and calculate its state data
            new_state_data = get_state_data_function(Solution(new_solution))
            if new_state_data is not None:  # Only proceed if the solution is valid
                new_profit = new_state_data['current_profit']
                # If the new solution is better, update best_operator and best_profit
                if new_profit > best_profit:
                    best_operator = FlipBlockOperator(list(indices_to_flip))
                    best_profit = new_profit

    # Step 4: Attempt swap refinement
    for item_in in items_in_knapsack:
        for item_out in feasible_items_to_add:
            # Create a potential new solution with the swap
            new_solution = current_solution.item_inclusion[:]
            new_solution[item_in], new_solution[item_out] = new_solution[item_out], new_solution[item_in]

            # Get the state data for the new solution
            new_state_data = get_state_data_function(Solution(new_solution))
            if new_state_data is not None:  # Only proceed if the solution is valid
                new_profit = new_state_data["current_profit"]
                if new_profit > best_profit:
                    best_operator = SwapOperator(item_in, item_out)
                    best_profit = new_profit

    # If an improvement was found via k-flip or swap, return it
    if best_operator is not None:
        return best_operator, {}

    # If no valid operator was found, return None
    return None, {}