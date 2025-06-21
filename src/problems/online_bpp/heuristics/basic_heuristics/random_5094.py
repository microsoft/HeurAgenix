from src.problems.online_bpp.components import Solution, AssignBinOperator, NewBinOperator
import random

def random_5094(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AssignBinOperator, dict]:
    """ Randomly assign the current item to one of the existing bins that have enough remaining capacity, 
    or place it in a new bin if no such bin exists.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - capacity (int): The capacity for each bin.
            - current_solution (Solution): The current solution object representing the bins and items.
            - current_item_size (int): The size of the current item to be packed.
            - remaining_capacity (list[int]): List of remaining capacities for each bin.
        problem_state["get_problem_state"] (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.

    Returns:
        An instance of AssignBinOperator or NewBinOperator, based on the decision made by the heuristic.
        An empty dictionary as no algorithm data is updated in this heuristic.
    """
    current_item_size = problem_state["current_item_size"]
    remaining_capacity = problem_state["remaining_capacity"]
    current_solution = problem_state["current_solution"]

    # If there is no current item to pack, return None
    if current_item_size is None:
        return None, {}

    feasible_bins = [i for i, capacity in enumerate(remaining_capacity) if capacity >= current_item_size]
    
    # If no feasible bins exist, create a new bin
    if not feasible_bins:
        operator = NewBinOperator()
    else:
        # Randomly select a feasible bin to assign the current item
        chosen_bin = random.choice(feasible_bins)
        operator = AssignBinOperator(chosen_bin)
    
    return operator, {}