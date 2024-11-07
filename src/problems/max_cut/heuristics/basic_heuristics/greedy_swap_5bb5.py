from src.problems.max_cut.components import Solution, SwapOperator

def greedy_swap_5bb5(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[SwapOperator, dict]:
    """
    Greedy Swap Heuristic for the Max Cut problem.
    Iteratively evaluates the delta in cut value for all possible single-node swaps between set A and set B,
    and performs the swap that leads to the highest increase in the cut value.
    If no swap improves the cut value, no operator is returned.

    Args:
        global_data (dict): Contains the global data of the graph.
            - "weight_matrix" (numpy.ndarray): A 2D array representing the weight between nodes.
        state_data (dict): Contains the current state information.
            - "current_solution" (Solution): The current solution of the Max Cut problem.
            - "current_cut_value" (int or float): The total weight of edges between set A and set B in the current solution.
        algorithm_data (dict): Not used in this heuristic.
        get_state_data_function (callable): Function to get the state data for a new solution.
        **kwargs: Additional hyperparameters, not used in this heuristic.

    Returns:
        SwapOperator: The operator that swaps a single node between sets to improve the cut value.
        dict: Empty dictionary as no algorithm data is updated.
    """
    current_solution = state_data['current_solution']
    weight_matrix = global_data['weight_matrix']
    best_increase = 0
    best_node = None

    # Precompute the sum of weights to and from each node to sets A and B
    weight_to_a = weight_matrix[:, list(current_solution.set_a)].sum(axis=1)
    weight_to_b = weight_matrix[:, list(current_solution.set_b)].sum(axis=1)

    # Evaluate all possible swaps to find the best one
    for node in range(len(weight_matrix)):
        if node in current_solution.set_a:
            # Calculate the delta in cut value for moving this node from A to B
            delta = weight_to_a[node] - weight_to_b[node]
        elif node in current_solution.set_b:
            # Calculate the delta in cut value for moving this node from B to A
            delta = weight_to_b[node] - weight_to_a[node]
        else:
            continue  # Skip if node is not in either set

        # Check if this swap improves the cut value
        if delta > best_increase:
            best_increase = delta
            best_node = node

    # If a beneficial swap was found, return the corresponding operator
    if best_node is not None:
        return SwapOperator([best_node]), {}
    else:
        return None, {}