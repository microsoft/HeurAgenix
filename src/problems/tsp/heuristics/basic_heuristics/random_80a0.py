from src.problems.tsp.components import Solution, AppendOperator
import random

def random_80a0(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AppendOperator, dict]:
    """ 
    Uniform-random constructive append for an open TSP path. At each iteration it uniformly samples one node from the unvisited set and appends it to the end of the current path, with no distance or marginal-cost evaluation. This distance-agnostic, memoryless move maximizes diversification and provides a high-entropy baseline for multi-start/portfolio hyper-heuristics. By appending only unvisited nodes it preserves a simple path and cannot create subtours; tour closure is deferred to subsequent operators. Invariant to symmetry, scaling, and sparsity of the distance matrix, and requires no algorithm_data. Computational cost per step is O(1) with constant memory; reproducibility is determined solely by the RNG state used by random.

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