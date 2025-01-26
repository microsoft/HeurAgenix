from src.problems.max_cut.components import *
import numpy as np

def highest_weight_edge_ca02(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertNodeOperator, dict]:
    """ Heuristic algorithm to maximize cut value by evaluating the impact of placing each unselected node into set A or B.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): A 2D array representing the weight between nodes.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current solution of the MaxCut problem.
            - "unselected_nodes" (set[int]): The set of unselected nodes.
            - "set_a" (set[int]): The set of nodes in partition A.
            - "set_b" (set[int]): The set of nodes in partition B.
        algorithm_data (dict): Not used in this algorithm.
        get_state_data_function (callable): Not used in this algorithm.

    Kwargs:
        k (int, optional): Frequency of applying swap operations. Default is 3.

    Returns:
        InsertNodeOperator: An operator that places an unselected node into one of the sets (A or B) to maximize the cut value.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary information
    weight_matrix = global_data["weight_matrix"]
    current_solution = state_data["current_solution"]
    unselected_nodes = state_data["unselected_nodes"]
    set_a = state_data["current_solution"].set_a
    set_b = state_data["current_solution"].set_b
    k = kwargs.get("k", 3)  # Frequency of swap operations, default is 3
    
    # If there are no unselected nodes, return None
    if not unselected_nodes:
        return None, {}

    # Initialize variables for the best node and its placement
    best_node = None
    best_target_set = None
    max_delta = float('-inf')

    # Evaluate each unselected node
    for node in unselected_nodes:
        # Calculate delta for placing the node in set A
        delta_a = sum(weight_matrix[node][other] for other in set_b) - sum(weight_matrix[node][other] for other in set_a)

        # Calculate delta for placing the node in set B
        delta_b = sum(weight_matrix[node][other] for other in set_a) - sum(weight_matrix[node][other] for other in set_b)

        # Determine the best placement for the current node
        if delta_b > delta_a and delta_b > max_delta:
            best_node = node
            best_target_set = 'B'
            max_delta = delta_b
        elif delta_a >= delta_b and delta_a > max_delta:
            best_node = node
            best_target_set = 'A'
            max_delta = delta_a

    # If no valid node is found, return None
    if best_node is None:
        return None, {}

    # Perform swap operation periodically (every k operations)
    if len(set_a) + len(set_b) % k == 0:
        best_swap_delta = float('-inf')
        best_swap_pair = None

        # Precompute weights to sets
        weight_to_a = weight_matrix[:, list(set_a)].sum(axis=1)
        weight_to_b = weight_matrix[:, list(set_b)].sum(axis=1)

        # Evaluate all possible swaps
        for i in set_a:
            for j in set_b:
                # Calculate delta for swapping nodes i and j
                delta = weight_to_a[i] - weight_to_a[j] + weight_to_b[j] - weight_to_b[i]
                if weight_matrix[i, j] != 0:
                    delta += 2 * weight_matrix[i, j]

                # Check if the swap improves the cut value
                if delta > best_swap_delta:
                    best_swap_delta = delta
                    best_swap_pair = (i, j)

        # If a beneficial swap is found, execute it
        if best_swap_pair and best_swap_delta > 0:
            i, j = best_swap_pair
            return SwapOperator(nodes=[i, j]), {}

    # Return the best node insertion operator
    return InsertNodeOperator(node=best_node, target_set=best_target_set), {}