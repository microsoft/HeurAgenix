from src.problems.online_bpp.components import Solution, AssignBinOperator, NewBinOperator
import random

def random_5094(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AssignBinOperator, dict]:
    """ Randomly assign the current item to one of the existing bins that have enough remaining capacity, 
    or place it in a new bin if no such bin exists.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - capacity (int): The capacity for each bin.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): The current solution object representing the bins and items.
            - current_item_size (int): The size of the current item to be packed.
            - remaining_capacity (list[int]): List of remaining capacities for each bin.
        get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.

    Returns:
        An instance of AssignBinOperator or NewBinOperator, based on the decision made by the heuristic.
        An empty dictionary as no algorithm data is updated in this heuristic.
    """
    current_item_size = state_data["current_item_size"]
    remaining_capacity = state_data["remaining_capacity"]
    current_solution = state_data["current_solution"]

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