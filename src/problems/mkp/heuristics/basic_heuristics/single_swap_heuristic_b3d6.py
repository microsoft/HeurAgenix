from src.problems.mkp.components import *

def single_swap_heuristic_b3d6(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[SwapOperator, dict]:
    """
    SingleSwapHeuristic tries to find the best single swap of items that increases the total profit
    of the knapsack without violating resource constraints.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weights" (numpy.array): A 2D array where each row represents the resource consumption of an item across all dimensions.
            - "profits" (numpy.array): The profit value associated with each item.
            - "capacities" (numpy.array): The maximum available capacity for each resource dimension.
            - "current_solution" (Solution): Current selection of items in the knapsack.
            - "current_profit" (float): Total profit of the current solution.
            - "current_weights" (numpy.array): The total resource consumption for each dimension in the current solution.
            - "items_in_knapsack" (list[int]): Indices of items currently in the knapsack.
            - "items_not_in_knapsack" (list[int]): Indices of items currently not in the knapsack.
            - "remaining_capacity" (numpy.array): The remaining capacity for each resource dimension after considering the items included in the current solution.
            - "get_problem_state" (callable): def validation_solution(solution: Solution) -> bool: The function to get the problem state for given solution without modify it.
        algorithm_data (dict): Not used in this heuristic.

    Returns:
        SwapOperator: The operator that defines the best swap action to take.
        dict: Empty dictionary as this heuristic does not update algorithm_data.
    """

    # Initialize variables
    best_swap_operator = None
    best_profit_increase = 0

    # Retrieve necessary data from global_data and state_data
    weights = problem_state["weights"]
    profits = problem_state["profits"]
    capacities = problem_state["capacities"]
    current_solution = problem_state["current_solution"]
    current_profit = problem_state["current_profit"]
    items_in_knapsack = problem_state["items_in_knapsack"]
    items_not_in_knapsack = problem_state["items_not_in_knapsack"]
    validation_solution = problem_state["validation_solution"]


    # Iterate over all pairs of items (one in the knapsack and one not in the knapsack)
    for item_in in items_in_knapsack:
        for item_out in items_not_in_knapsack:
            # Create a potential new solution with the swap
            new_solution = current_solution.item_inclusion[:]
            new_solution[item_in], new_solution[item_out] = new_solution[item_out], new_solution[item_in]
            
            # Get the state data for the new solution
            if validation_solution(Solution(new_solution)):
                new_problem_state = problem_state["get_problem_state"](Solution(new_solution))
                if new_problem_state:
                    # Calculate the profit increase for this swap
                    new_profit = new_problem_state["current_profit"]
                    profit_increase = new_profit - current_profit

                    # Check if this swap is the best so far and update if necessary
                    if profit_increase > best_profit_increase and all(new <= cap for new, cap in zip(new_problem_state["current_weights"], capacities)):
                        best_profit_increase = profit_increase
                        best_swap_operator = SwapOperator(item_in, item_out)

    # If a valid swap that improves the profit was found, return the corresponding SwapOperator
    if best_swap_operator is not None:
        return best_swap_operator, {}
    else:
        # No swap found that improves the solution, return None
        return None, {}