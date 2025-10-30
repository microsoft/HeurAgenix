from src.problems.max_cut.components import *

def highest_delta_edge_9f66(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertEdgeOperator, dict]:
    """
    Greedy pairwise best-improvement constructive step. At each call, considers all unordered pairs of unselected nodes and both orientations (i→A,j→B vs i→B,j→A), selecting the orientation that maximizes the immediate cut gain. Incremental gain model uses precomputed affiliation sums to the current partition: delta(i→A,j→B)=sum_w(i,B)+sum_w(j,A)+w(i,j); delta(i→B,j→A)=sum_w(i,A)+sum_w(j,B)+w(i,j). The global argmax over all unselected pairs is returned as an InsertEdgeOperator with nodes ordered to match the chosen orientation. If exactly one node remains unselected, it is inserted into A without optimization. Ties are implicitly broken by first encounter (strict > update). Works on weighted graphs without assuming symmetry (interprets the weight matrix as given). Time: O(n(|A|+|B|)) to precompute affiliation sums plus O(|U|^2) to scan pairs; memory: O(n). Suitable for constructive/repair phases; emphasizes joint (two-node) gain over single-node myopic insertions.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): A 2D array representing the weight between nodes.
            - "current_solution" (Solution): The current partition of the graph into sets A and B.
            - "selected_nodes" (set[int]): The set of selected nodes.

    Returns:
        InsertEdgeOperator: Operator to insert the nodes of the edge into the appropriate sets.
        dict: Empty dictionary as no algorithm data is updated.
    """
    weight_matrix = problem_state["weight_matrix"]
    current_solution = problem_state["current_solution"]
    selected_nodes = problem_state["selected_nodes"]
    unselected_nodes = problem_state["unselected_nodes"]

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