from src.problems.max_cut.components import *

def highest_delta_edge_9f66(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertEdgeOperator, dict]:
    """Selects the unselected edge that maximizes the increase in cut weight when added to the solution.

    Args:
        global_data (dict): Contains global information about the graph.
            - "weight_matrix" (numpy.ndarray): A 2D array representing the weight between nodes.
        state_data (dict): Contains the current state of the solution.
            - "current_solution" (Solution): The current partition of the graph into sets A and B.
            - "selected_nodes" (set[int]): The set of selected nodes.

    Returns:
        InsertEdgeOperator: Operator to insert the nodes of the edge into the appropriate sets.
        dict: Empty dictionary as no algorithm data is updated.
    """
    weight_matrix = global_data["weight_matrix"]
    current_solution = state_data["current_solution"]
    selected_nodes = state_data["selected_nodes"]
    unselected_nodes = state_data["unselected_nodes"]

    best_delta = -float('inf')
    best_edge = None
    best_set_a = None
    best_set_b = None

    if len(unselected_nodes) == 1:
        return InsertNodeOperator(node=next(iter(unselected_nodes)), target_set='A'), {}

    # Precompute the sum of weights connected to each node for both sets A and B
    delta_set_a = [sum(weight_matrix[i][other] for other in current_solution.set_a) for i in range(len(weight_matrix))]
    delta_set_b = [sum(weight_matrix[i][other] for other in current_solution.set_b) for i in range(len(weight_matrix))]

    # Iterate over all pairs of nodes to find the best edge to add to the solution
    for i in range(len(weight_matrix)):
        if i in selected_nodes:
            continue  # Skip if node i is selected

        for j in range(i + 1, len(weight_matrix)):
            if j in selected_nodes:
                continue  # Skip if node j is selected

            # Calculate the delta for both possible insertions and choose the best one
            delta_a_to_b = delta_set_b[i] + delta_set_a[j] + weight_matrix[i][j]
            delta_b_to_a = delta_set_a[i] + delta_set_b[j] + weight_matrix[i][j]

            if delta_a_to_b > delta_b_to_a:
                delta = delta_a_to_b
                set_a = 'A'
                set_b = 'B'
            else:
                delta = delta_b_to_a
                set_a = 'B'
                set_b = 'A'

            # Update the best edge if the current delta is greater than the best delta found so far
            if delta > best_delta:
                best_delta = delta
                best_edge = (i, j)
                best_set_a = set_a
                best_set_b = set_b

    # If no edge is found, return None
    if best_edge is None:
        return None, {}

    # Create the operator to insert the nodes of the best edge into the chosen sets
    node_1, node_2 = best_edge
    if best_set_a == 'A':
        operator = InsertEdgeOperator(node_1=node_1, node_2=node_2)
    else:
        operator = InsertEdgeOperator(node_1=node_2, node_2=node_1)

    return operator, {}