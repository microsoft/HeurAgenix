from src.problems.mkp.components import *
import numpy as np

def greedy_by_profit_58f2(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, profit_threshold_percentile: float = 0.8, reward_weight: float = 1.0, **kwargs) -> tuple[BaseOperator, dict]:
    """Greedy heuristic for the Multidimensional Knapsack Problem with enhanced opportunity cost scoring and swap refinement.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "profits" (numpy.array): The profit value associated with each item.
            - "weights" (list of lists): A 2D list where each sublist represents the resource consumption of an item across all dimensions.
            - "capacities" (numpy.array): The maximum available capacity for each resource dimension.
            - "resource_num" (int): The number of resource dimensions.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "remaining_capacity" (numpy.array): The remaining capacity for each resource dimension.
            - "items_in_knapsack" (list[int]): A list of item indices that are currently included in the knapsack.
            - "feasible_items_to_add" (list[int]): A list of item indices that can be added without violating constraints.
            - "current_solution" (Solution): The current solution object.
            - "current_profit" (float): The current total profit of the solution.
        algorithm_data (dict, optional): The algorithm dictionary for current algorithm only. In this algorithm, no specific data is necessary.
        get_state_data_function (callable, optional): The function receives the new solution as input and return the state dictionary for the new solution, and it will not modify the original solution.
        profit_threshold_percentile (float, optional): The percentile threshold for the profit-to-weight ratio, defaults to 0.8.
        reward_weight (float, optional): The weight to apply as a reward factor for known optimal solutions, defaults to 1.0.

    Returns:
        BaseOperator: The operator to apply to the current solution (e.g., AddOperator, RemoveOperator, FlipBlockOperator, SwapOperator).
        dict: An empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data from global_data
    profits = global_data["profits"]
    weights = global_data["weights"]
    resource_num = global_data["resource_num"]

    # Extract necessary data from state_data
    remaining_capacity = state_data["remaining_capacity"]
    items_in_knapsack = state_data["items_in_knapsack"]
    feasible_items_to_add = state_data["feasible_items_to_add"]
    current_solution = state_data["current_solution"]
    current_profit = state_data["current_profit"]

    # Calculate profit-to-weight ratios and apply percentile filter
    profit_to_weight_ratios = [(profits[item] / (sum(weights[res][item] for res in range(resource_num)) + 1e-9), item) for item in feasible_items_to_add]
    
    if not profit_to_weight_ratios:
        return None, {}

    threshold_value = np.percentile([ratio for ratio, _ in profit_to_weight_ratios], profit_threshold_percentile * 100)
    filtered_items = [item for ratio, item in profit_to_weight_ratios if ratio >= threshold_value]

    # Opportunity Cost Scoring with reward mechanism
    best_operator = None
    best_score = float('-inf')

    for item in filtered_items:
        profit_to_weight_ratio = profits[item] / (sum(weights[res][item] for res in range(resource_num)) + 1e-9)
        capacity_adjustment = np.min(
            [remaining_capacity[res] / (weights[res][item] + 1e-9) for res in range(resource_num) if weights[res][item] > 0],
            initial=float('inf')
        )
        opportunity_cost = max(
            [profits[other_item] / (sum(weights[res][other_item] for res in range(resource_num)) + 1e-9)
             for other_item in filtered_items if other_item != item],
            default=0
        )
        # Reward mechanism for aligning with known optimal solutions
        reward = reward_weight if item in algorithm_data.get("optimal_solution_items", []) else 1.0
        score = (profit_to_weight_ratio * capacity_adjustment - opportunity_cost) * reward

        if score > best_score:
            best_operator = AddOperator(item)
            best_score = score

    # If a valid AddOperator is found, return it
    if best_operator is not None:
        return best_operator, {}

    # Swap Refinement using a basic swap heuristic
    best_profit = current_profit
    for item_in in items_in_knapsack:
        for item_out in feasible_items_to_add:
            # Create potential new solution by swapping items
            new_solution = current_solution.item_inclusion[:]
            new_solution[item_in], new_solution[item_out] = new_solution[item_out], new_solution[item_in]

            # Validate and evaluate new solution
            new_state_data = get_state_data_function(Solution(new_solution))
            if new_state_data and new_state_data["current_profit"] > best_profit:
                return SwapOperator(item_in, item_out), {}

    # If no operator improves the solution, return None
    return None, {}