from src.problems.max_cut.components import *

def balanced_cut_21d5(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, max_iterations: int = 100) -> tuple[InsertNodeOperator, dict]:
    """
    Balanced Cut heuristic for the Max Cut problem. This heuristic tries to balance the number of nodes in sets A and B
    while maximizing the cut value. It iteratively adds nodes to the smaller set until a balanced state is reached or
    the maximum number of iterations is exceeded.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_nodes" (int): The total number of vertices in the graph.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current solution of the Max Cut problem.
            - "unselected_nodes" (set[int]): The set of nodes that have not been selected into either set A or set B.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this heuristic.
        get_state_data_function (callable): Function to get the state data for a new solution.
        max_iterations (int): The maximum number of iterations to perform. Defaults to 100.

    Returns:
        InsertNodeOperator: The operator to insert a node into the smaller set.
        dict: Empty dictionary as no algorithm data is updated.
    """
    current_solution = state_data['current_solution']
    unselected_nodes = state_data['unselected_nodes']

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