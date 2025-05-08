from src.problems.max_cut.components import *

def highest_weight_edge_eb0c(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertEdgeOperator, dict]:
    """
    Selects an edge with the highest weight where both nodes are unselected and adds each node to opposite sets,
    choosing the set that maximizes the cut value increase for each node.

    Args:
        global_data (dict): Contains the global information about the graph.
            - "weight_matrix" (numpy.ndarray): A 2D array representing the weight between nodes.

        state_data (dict): Contains the current state of the solution.
            - "current_solution" (Solution): The current solution of the MaxCut problem.
            - "unselected_nodes" (set[int]): The set of unselected nodes.

    Returns:
        InsertEdgeOperator: Operator to insert the selected edge into the solution, with each node added to the set
        that maximizes the cut value increase.
        dict: Empty dictionary as no algorithm data is updated.
    """
    
    # Extract necessary information from global_data and state_data
    weight_matrix = global_data["weight_matrix"]
    current_solution = state_data["current_solution"]
    unselected_nodes = state_data["unselected_nodes"]
    
    # Initialize variables to track the highest weight and the corresponding edge
    highest_weight = float('-inf')
    selected_edge = None
    
    if len(unselected_nodes) == 1:
        return InsertNodeOperator(node=next(iter(unselected_nodes)), target_set='A'), {}

    # Iterate over the weight_matrix to find the unselected edge with the highest weight
    for i in unselected_nodes:
        for j in unselected_nodes:
            if i == j:
                continue
            # Ensure both nodes are unselected and the edge weight is higher than the current highest
            if weight_matrix[i][j] > highest_weight:
                highest_weight = weight_matrix[i][j]
                selected_edge = (i, j)

    # If no edge is found, return None
    if selected_edge is None:
        return None, {}

    # Calculate the potential increase in cut value for adding each node to set A and set B
    node_1, node_2 = selected_edge
    potential_increase_a1 = sum(weight_matrix[node_1][other] for other in current_solution.set_b)
    potential_increase_b1 = sum(weight_matrix[node_1][other] for other in current_solution.set_a)
    potential_increase_a2 = sum(weight_matrix[node_2][other] for other in current_solution.set_b)
    potential_increase_b2 = sum(weight_matrix[node_2][other] for other in current_solution.set_a)

    # Create an operator to insert the nodes of the selected edge into the appropriate sets
    if node_1 in current_solution.set_a or node_2 in current_solution.set_b:
        op = InsertEdgeOperator(node_1=node_1, node_2=node_2)
    elif node_1 in current_solution.set_b or node_2 in current_solution.set_a:
        op = InsertEdgeOperator(node_1=node_2, node_2=node_1)
    elif potential_increase_a1 >= potential_increase_b1:
        op = InsertEdgeOperator(node_1=node_1, node_2=node_2)
    else:
        op = InsertEdgeOperator(node_1=node_2, node_2=node_1)

    # Return the operator along with an empty dictionary as no algorithm data is updated
    return op, {}