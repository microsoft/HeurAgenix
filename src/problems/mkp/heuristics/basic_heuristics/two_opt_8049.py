from src.problems.mkp.components import *

def two_opt_8049(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[SwapOperator, dict]:
    """
    Two-Opt heuristic that iteratively evaluates potential swaps of two items' inclusion statuses in the knapsack. 
    It retains the swap if it leads to a solution with a higher profit without violating any resource constraints.

    Args:
        global_data (dict): The global data dict containing the problem's constants. In this algorithm, the following items are necessary:
            - "weights" (numpy.array): A 2D array where each row represents the resource consumption of an item across all dimensions.
            - "capacities" (numpy.array): The maximum available capacity for each resource dimension.
            
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current solution instance.
            - "items_in_knapsack" (list[int]): List of item indices currently in the knapsack.
            - "items_not_in_knapsack" (list[int]): List of item indices not in the knapsack.
            - "current_weights" (numpy.array): The total resource consumption for each dimension in the current solution.            - "current_profit" (float): The total profit of items in the current solution.

        algorithm_data (dict): The algorithm dictionary for current algorithm only. This algorithm does not use algorithm_data.
        get_state_data_function (callable): Function that takes a new Solution instance and returns the corresponding state data without modifying the original solution.

    Returns:
        SwapOperator: The operator to apply to the current solution if an improvement is found.
        dict: An empty dictionary as this algorithm does not update the algorithm data.
    """
    
    best_operator = None
    best_profit = state_data["current_profit"]
    items_in_knapsack = state_data["items_in_knapsack"]
    items_not_in_knapsack = state_data["items_not_in_knapsack"]
    weights = global_data["weights"]
    capacities = global_data["capacities"]

    # Iterate through pairs of items: one in the knapsack and one not in the knapsack
    for item_in in items_in_knapsack:
        for item_out in items_not_in_knapsack:
            # Create a new solution by swapping the two items
            new_solution = state_data["current_solution"].item_inclusion[:]
            new_solution[item_in], new_solution[item_out] = new_solution[item_out], new_solution[item_in]
            temp_solution = Solution(new_solution)
            
            # Get the state data for the new solution
            new_state_data = get_state_data_function(temp_solution)
            
            # If the new solution is valid and has a higher profit, store it as the best solution
            if new_state_data and new_state_data["current_profit"] > best_profit:
                best_operator = SwapOperator(item_in, item_out)
                best_profit = new_state_data["current_profit"]
    
    # Return the best swap operator found and an empty dictionary for algorithm data
    return (best_operator, {}) if best_operator else (None, {})
