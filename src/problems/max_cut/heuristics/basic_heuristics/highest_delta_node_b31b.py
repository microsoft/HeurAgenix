from src.problems.max_cut.components import *

def highest_delta_node_b31b(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertNodeOperator, dict]:
    """
    This heuristic selects the unselected node that, when added to one of the sets (A or B),
    would result in the largest increase in the total cut weight. The node is then inserted into
    that set which maximizes the cut weight.

    Args:
        global_data (dict): Contains information about the graph.
            - "weight_matrix" (numpy.ndarray): A 2D array representing the weight between nodes.
        state_data (dict): Contains information about the current state of the solution.
            - "current_solution" (Solution): The current partition of the graph into sets A and B.
            - "unselected_count" (int): The number of nodes not yet selected into either set A or B.
            - "unselected_nodes" (set[int]): The set of unselected nodes.
        algorithm_data (dict): Not used in this heuristic.
        get_state_data_function (callable): Function to get state data for a new solution.

    Returns:
        InsertNodeOperator: The operator to insert the node into the appropriate set.
        dict: Empty dictionary as this algorithm doesn't update the algorithm data.
    """

    # Extract necessary information from global_data and state_data
    weight_matrix = global_data["weight_matrix"]
    current_solution = state_data["current_solution"]
    unselected_nodes = state_data["unselected_nodes"]

    # Initialize variables to keep track of the best node and delta
    best_node = None
    best_delta = -float('inf')

    # Iterate over all unselected nodes to find the one with the highest delta
    for node in unselected_nodes:
        delta_a = sum(weight_matrix[node, other] for other in current_solution.set_b)
        delta_b = sum(weight_matrix[node, other] for other in current_solution.set_a)

        # Check if placing the node in set A or B gives a better delta
        if delta_a > best_delta or delta_b > best_delta:
            best_delta = max(delta_a, delta_b)
            best_node = node
            target_set = 'A' if delta_a > delta_b else 'B'

    # If no suitable node is found (all nodes are already selected), return None
    if best_node is None:
        return None, {}

    # Create the operator to insert the best node into the chosen set
    operator = InsertNodeOperator(best_node, target_set)

    # Return the operator and an empty dictionary as no algorithm data is updated
    return operator, {}