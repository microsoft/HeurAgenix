from src.problems.max_cut.components import *

def balanced_cut_21d5(problem_state: dict, algorithm_data: dict, max_iterations: int = 100) -> tuple[InsertNodeOperator, dict]:
    """
    Balance-driven single-node insertion. Selects the first available unselected node (order-dependent via set-to-list conversion) and inserts it into the smaller partition; ties are resolved in favor of set A, inducing a slight bias. Ignores edge weights and cut gain entirely, focusing solely on cardinality balance; thus it is constructive/feasibility-oriented and may not improve the cut value. Operates one node per call, enabling gradual balancing across repeated invocations. Inputs effectively used: current_solution and unselected_nodes; total_nodes and max_iterations are unused. Applicable to weighted or unweighted graphs without structural assumptions. Time complexity O(1); constant extra memory.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "total_nodes" (int): The total number of vertices in the graph.
            - "current_solution" (Solution): The current solution of the Max Cut problem.
            - "unselected_nodes" (set[int]): The set of nodes that have not been selected into either set A or set B.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this heuristic.
        max_iterations (int): The maximum number of iterations to perform. Defaults to 100.

    Returns:
        InsertNodeOperator: The operator to insert a node into the smaller set.
        dict: Empty dictionary as no algorithm data is updated.
    """
    current_solution = problem_state['current_solution']
    unselected_nodes = problem_state['unselected_nodes']

    # If there are no unselected nodes left, return None.
    if not unselected_nodes:
        return None, {}

    set_a_count = len(current_solution.set_a)
    set_b_count = len(current_solution.set_b)
    # Select a node to insert
    node_to_insert = list(unselected_nodes)[0]
    # Determine which set is smaller and add the node to it
    if set_a_count <= set_b_count:
        target_set = 'A'
    else:
        target_set = 'B'
    # Create the operator to insert the node
    operator = InsertNodeOperator(node=node_to_insert, target_set=target_set)
    return operator, {}