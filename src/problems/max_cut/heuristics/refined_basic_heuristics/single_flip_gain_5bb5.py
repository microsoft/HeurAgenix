from src.problems.max_cut.components import *
import numpy as np

def single_flip_gain_5bb5(problem_state: dict, algorithm_data: dict, accept_zero_gain: bool=True, fallback_policy: str='balance', **kwargs) -> tuple[SwapOperator, dict]:
    """Single-node flip using per-node gain with an optional non-degrading fallback.
    
    This heuristic computes, for each currently assigned vertex, the change in cut value (delta)
    if the vertex is flipped to the opposite partition. It selects the node with the largest
    gain (delta) under an improvement criterion. If a strictly positive gain is unavailable,
    it optionally applies a fallback policy to avoid returning None:
      - 'balance': choose the node from the larger set (A or B) that has the highest delta,
        which tends to balance the partition sizes while making the best available move.
      - 'max_delta': choose the globally highest delta regardless of set, even if non-positive.
    
    Gains are computed from row sums of the weight matrix (supports weighted/directed graphs).
    Precomputation aggregates each node's total weight to set A and to set B; flipping a node
    in A yields delta = sum_to_A - sum_to_B, and flipping a node in B yields delta = sum_to_B - sum_to_A.
    Ties are resolved by first-seen scan order. No iterative updates or pairwise swaps are performed.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): A square 2D array (n x n) with edge weights; rows used for outgoing totals.
            - "current_solution" (Solution): The current MaxCut partition containing set_a and set_b.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this heuristic.
        accept_zero_gain (bool): Whether to treat zero-gain flips as acceptable improvements (delta >= 0). Default is True.
        fallback_policy (str): Policy when no acceptable improvement exists. One of {"balance", "max_delta"}.
                               - "balance": prefer flipping from the larger set; pick the node with highest delta in that set.
                               - "max_delta": pick the globally highest delta (may be non-positive).
                               Default is "balance".

    Returns:
        SwapOperator: The operator that swaps a single node between sets to improve the cut value, or applies the chosen fallback if no positive gain is found.
        dict: Empty dictionary as no algorithm data is updated.
    """
    current_solution = problem_state['current_solution']
    weight_matrix = problem_state['weight_matrix']

    # If no nodes are assigned, a flip is impossible.
    if not current_solution.set_a and not current_solution.set_b:
        return None, {}

    n = weight_matrix.shape[0]
    set_a_list = list(current_solution.set_a)
    set_b_list = list(current_solution.set_b)

    # Precompute per-node sums to A and B (handle empty sets safely).
    weight_to_a = weight_matrix[:, set_a_list].sum(axis=1) if set_a_list else np.zeros(n)
    weight_to_b = weight_matrix[:, set_b_list].sum(axis=1) if set_b_list else np.zeros(n)

    # Track best acceptable improvement and global best for fallback.
    best_improve_node = None
    best_improve_delta = -float('inf')  # Updated only when improvement criterion is satisfied.

    global_best_node = None
    global_best_delta = -float('inf')

    # Scan all assigned nodes and evaluate flip gains.
    for node in range(n):
        if node in current_solution.set_a:
            delta = weight_to_a[node] - weight_to_b[node]
        elif node in current_solution.set_b:
            delta = weight_to_b[node] - weight_to_a[node]
        else:
            continue  # Node not yet assigned; SwapOperator cannot flip it.

        # Update global best (for fallback).
        if delta > global_best_delta or (global_best_node is None and delta == global_best_delta):
            global_best_delta = delta
            global_best_node = node

        # Check improvement criterion.
        is_improvement = (delta > 0.0) if not accept_zero_gain else (delta >= 0.0)
        if is_improvement and (delta > best_improve_delta or best_improve_node is None):
            best_improve_delta = delta
            best_improve_node = node

    # If an acceptable improvement exists, use it.
    if best_improve_node is not None:
        return SwapOperator([best_improve_node]), {}

    # Fallback selection to avoid returning None.
    if fallback_policy == 'balance':
        # Determine larger set (break ties in favor of A).
        larger_set = 'A' if len(current_solution.set_a) >= len(current_solution.set_b) else 'B'
        best_in_larger = None
        best_delta_in_larger = -float('inf')

        # Scan only nodes in the larger set to pick the highest delta.
        candidate_nodes = current_solution.set_a if larger_set == 'A' else current_solution.set_b
        for node in candidate_nodes:
            if node in current_solution.set_a:
                delta = weight_to_a[node] - weight_to_b[node]
            else:
                delta = weight_to_b[node] - weight_to_a[node]

            if delta > best_delta_in_larger or (best_in_larger is None and delta == best_delta_in_larger):
                best_delta_in_larger = delta
                best_in_larger = node

        if best_in_larger is not None:
            return SwapOperator([best_in_larger]), {}

        # If somehow larger set is empty, fall back to global best.
        if global_best_node is not None:
            return SwapOperator([global_best_node]), {}

        # No valid flip candidates found.
        return None, {}

    else:  # 'max_delta' or any unrecognized policy: fallback to global max delta
        if global_best_node is not None:
            return SwapOperator([global_best_node]), {}
        return None, {}