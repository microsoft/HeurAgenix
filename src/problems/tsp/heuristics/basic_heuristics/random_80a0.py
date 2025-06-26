from src.problems.tsp.components import Solution, AppendOperator
import random

def random_80a0(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AppendOperator, dict]:
    """ Implements the random append heuristic for the TSP problem. 
    At each step, randomly select an unvisited node and append it to the current solution.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - unvisited_nodes (list[int]): A list of integers representing the IDs of nodes that have not yet been visited.

    Returns:
        AppendOperator: An operator that appends the selected node to the current solution.
        dict: The updated algorithm dictionary. In this case, it is empty as no additional data is required for future iterations.
    """
    # If there are no unvisited nodes left, return None and an empty dict
    if not problem_state["unvisited_nodes"]:
        return None, {}

    # Randomly select an unvisited node
    selected_node = random.choice(problem_state["unvisited_nodes"])

    # Create an AppendOperator with the selected node
    operator = AppendOperator(node=selected_node)

    return operator, {}