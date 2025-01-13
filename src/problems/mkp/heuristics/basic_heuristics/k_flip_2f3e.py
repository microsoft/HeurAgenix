from src.problems.mkp.components import *
from itertools import combinations
import random

def k_flip_2f3e(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, k: int = 2) -> tuple[FlipBlockOperator, dict]:
    """ K-flip heuristic that flips the inclusion status of k items.

    Args:
        global_data (dict): Global data needed for the heuristic.
        state_data (dict): Current state data needed to evaluate the flips.
        algorithm_data (dict): Not used in this algorithm.
        get_state_data_function (callable): Function to calculate state data for a new solution.
        k (int): The number of items to flip. Defaults to 2, can be increased as needed.

    Returns:
        FlipBlockOperator: The operator that flips k items if it results in a valid and improved solution.
        dict: Empty dictionary as the algorithm data is not updated.
    """
    # Extract necessary data from global_data
    item_num = global_data['item_num']
    all_indices = range(item_num)

    # Extract necessary data from state_data
    current_solution = state_data['current_solution']
    current_profit = state_data['current_profit']
    validation_solution = state_data['validation_solution']

    # Generate all possible combinations of k indices
    all_combinations = list(combinations(all_indices, k))
    random.shuffle(all_combinations)  # Randomize the order to avoid bias

    # Initialize best operator and corresponding profit to current state
    best_operator = None
    best_profit = current_profit

    # Iterate over all combinations and evaluate the flip
    for indices_to_flip in all_combinations:
        # Generate a new solution by flipping the k items
        new_solution = current_solution.item_inclusion[:]
        for index in indices_to_flip:
            new_solution[index] = not new_solution[index]

        # Check if the new solution is valid and calculate its state data
        if validation_solution(Solution(new_solution)):  # Only proceed if the solution is valid
            new_state_data = get_state_data_function(Solution(new_solution))
            new_profit = new_state_data['current_profit']

            # If the new solution is better, update best_operator and best_profit
            if new_profit > best_profit:
                best_operator = FlipBlockOperator(list(indices_to_flip))
                best_profit = new_profit

    # Return the best operator found and an empty dictionary as no algorithm data is updated
    return (best_operator, {}) if best_operator is not None else (None, {})
