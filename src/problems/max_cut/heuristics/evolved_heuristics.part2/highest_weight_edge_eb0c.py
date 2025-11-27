from src.problems.max_cut.components import *

def highest_weight_edge_eb0c(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertEdgeOperator, dict]:
    """
    Greedy constructive seeding by globally heaviest unselected edge. Selects the edge (i,j) with maximum weight among unselected node pairs, independent of current partition gains. Orientation is decided solely from node_1’s marginal gain against the current sets (sum to set_B vs set_A); node_2 is forced to the opposite set to ensure the chosen edge contributes to the cut. This yields a best-by-edge-weight choice, not a best-improvement-by-cut-gain choice, and may accept non-improving steps relative to the current cut. Single-node remainder is handled by inserting that node into set A without evaluation, introducing a deterministic bias in the terminal step. Ties are resolved by matrix scan order (no secondary criterion). Supports asymmetric weight matrices since all gains are computed from the provided directed weights. Time complexity: O(|U|^2) to find the edge plus O(|A|+|B|) to compute the orientation; O(1) additional memory.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): A 2D array representing the weight between nodes.
            - "current_solution" (Solution): The current solution of the MaxCut problem.
            - "unselected_nodes" (set[int]): The set of unselected nodes.

    Returns:
        InsertEdgeOperator: Operator to insert the selected edge into the solution, with each node added to the set
        that maximizes the cut value increase.
        dict: Empty dictionary as no algorithm data is updated.
    """
    
    # Extract necessary information from problem_state
    weight_matrix = problem_state["weight_matrix"]
    current_solution = problem_state["current_solution"]
    unselected_nodes = problem_state["unselected_nodes"]
    
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