from src.problems.tsp.components import Solution, AppendOperator
import random

def random_80a0(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AppendOperator, dict]:
    """ Implements the random append heuristic for the TSP problem. 
    At each step, randomly select an unvisited node and append it to the current solution.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - unvisited_nodes (list[int]): A list of integers representing the IDs of nodes that have not yet been visited.
        get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.

    Returns:
        AppendOperator: An operator that appends the selected node to the current solution.
        dict: The updated algorithm dictionary. In this case, it is empty as no additional data is required for future iterations.
    """
    # If there are no unvisited nodes left, return None and an empty dict
    if not state_data["unvisited_nodes"]:
        return None, {}

    # Randomly select an unvisited node
    selected_node = random.choice(state_data["unvisited_nodes"])

    # Create an AppendOperator with the selected node
    operator = AppendOperator(node=selected_node)

    return operator, {}