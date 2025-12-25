from src.problems.max_cut.components import *

def highest_delta_node_b31b(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertNodeOperator, dict]:
    """
    Greedy global best-improvement insertion for partial MaxCut partitions. For each unassigned node, evaluates both placements: ΔA = sum of weights from the node to current set B; ΔB = sum of weights from the node to current set A. Selects the node and target set with the largest Δ, performing an argmax over all (node, target_set) pairs. Strict-improvement updating: a candidate replaces the incumbent only if its Δ exceeds the current best; equal-Δ ties do not update (first-best retention). When ΔA == ΔB and the pair triggers an update (both strictly > current best), the tie breaks to set B. With an empty partition, the first inserted node goes to set B. Not restricted to positive gains; if all Δ are negative, it inserts the least-worsening node, making it suitable for constructive/repair phases rather than strict local improvement. Uses directed outgoing weights weight_matrix[node, other]; for asymmetric matrices it optimizes the node’s outgoing contribution to the cut. Time complexity: O(|unselected| × (|A| + |B|)); O(1) extra memory.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): A 2D array representing the weight between nodes.
            - "current_solution" (Solution): The current partition of the graph into sets A and B.
            - "unselected_count" (int): The number of nodes not yet selected into either set A or B.
            - "unselected_nodes" (set[int]): The set of unselected nodes.
        algorithm_data (dict): Not used in this heuristic.
        problem_state["get_problem_state"] (callable): Function to get state data for a new solution.

    Returns:
        InsertNodeOperator: The operator to insert the node into the appropriate set.
        dict: Empty dictionary as this algorithm doesn't update the algorithm data.
    """

    # Extract necessary information from problem_state
    weight_matrix = problem_state["weight_matrix"]
    current_solution = problem_state["current_solution"]
    unselected_nodes = problem_state["unselected_nodes"]

    # Initialize variables to keep track of the best node and delta
    best_node = None
    best_delta = -float('inf')
    
    import numpy as np
    
    # Convert sets to lists for indexing
    unselected_list = list(unselected_nodes)
    if not unselected_list:
        return None, {}
        
    set_a_list = list(current_solution.set_a)
    set_b_list = list(current_solution.set_b)
    
    # Vectorized calculation
    # Calculate weights from all unselected nodes to set A and set B at once
    # weight_matrix[unselected_list][:, set_a_list] gives a submatrix of weights
    # Summing along axis 1 gives the total weight to the set for each unselected node
    
    if set_a_list:
        weights_to_a = weight_matrix[unselected_list][:, set_a_list].sum(axis=1)
    else:
        weights_to_a = np.zeros(len(unselected_list))
        
    if set_b_list:
        weights_to_b = weight_matrix[unselected_list][:, set_b_list].sum(axis=1)
    else:
        weights_to_b = np.zeros(len(unselected_list))
    
    # delta_a is gain if moved to A (sum of weights to B)
    # delta_b is gain if moved to B (sum of weights to A)
    # Note: The original code logic was:
    # delta_a = sum(weight_matrix[node, other] for other in current_solution.set_b) -> This is gain if put in A?
    # Wait, MaxCut objective is to maximize edges BETWEEN sets.
    # If I put node in A, the cut edges are those connecting to B. So gain is sum(weights to B).
    # Yes, original code: delta_a = sum(... set_b). Correct.
    
    deltas_a = weights_to_b
    deltas_b = weights_to_a
    
    # Find max gain for each node
    max_gains = np.maximum(deltas_a, deltas_b)
    best_idx = np.argmax(max_gains)
    
    best_gain = max_gains[best_idx]
    best_node = unselected_list[best_idx]
    
    # Determine target set
    # If deltas_a[best_idx] > deltas_b[best_idx], then target is A
    if deltas_a[best_idx] > deltas_b[best_idx]:
        target_set = 'A'
    else:
        target_set = 'B'

    # Create the operator to insert the best node into the chosen set
    operator = InsertNodeOperator(best_node, target_set)

    # Return the operator and an empty dictionary as no algorithm data is updated
    return operator, {}
