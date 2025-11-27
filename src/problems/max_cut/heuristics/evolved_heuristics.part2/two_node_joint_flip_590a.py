from src.problems.max_cut.components import *

def two_node_joint_flip_590a(problem_state: dict, algorithm_data: dict, min_gain_threshold: float=1e-12, early_stop_first_improvement: bool=False, **kwargs) -> tuple[SwapOperator, dict]:
    """Two-node joint flip local search for MaxCut (pair swap across partition). Scans all cross-set pairs (i in A, j in B)
    and evaluates the simultaneous flip of both nodes to the opposite sets. The net cut-change is:
        Δ(i,j) = (Σ_A w(i,·) − Σ_B w(i,·)) + (Σ_B w(j,·) − Σ_A w(j,·)) + 2·w(i,j),
    where the +2·w(i,j) corrects the double subtraction of the cross edge (i,j) that occurs when summing single-node flips.
    This heuristic supports either first-improvement (early stop) or best-improvement (global best pair) acceptance.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): Symmetric, non-negative adjacency/weight matrix W of shape (n, n).
            - "current_solution" (Solution): Current partition with disjoint sets 'set_a' and 'set_b'.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this heuristic.
        min_gain_threshold (float): Strict-positivity threshold to accept an improvement; filters numerical noise around zero.
            Default is 1e-12. Must be >= 0.
        early_stop_first_improvement (bool): If True, returns the first pair with Δ > min_gain_threshold (first-improvement).
            If False, scans all pairs and returns the globally best improving pair (best-improvement). Default is False.

    Returns:
        SwapOperator: Operator that flips both selected nodes simultaneously (i from A->B and j from B->A). If no improving pair
            exists or one set is empty, returns None.
        dict: Empty dictionary; this heuristic does not update algorithm_data.

    """
    # Extract required state
    current_solution = problem_state.get('current_solution')
    weight_matrix = problem_state.get('weight_matrix')

    # Defensive checks: require both sets non-empty to form pairs
    set_a = current_solution.set_a
    set_b = current_solution.set_b
    if not set_a or not set_b:
        return None, {}

    # Precompute side sums for each node:
    # weight_to_a[u] = sum_{v in A} W[u,v]
    # weight_to_b[u] = sum_{v in B} W[u,v]
    idx_a = list(set_a)
    idx_b = list(set_b)
    weight_to_a = weight_matrix[:, idx_a].sum(axis=1) if idx_a else weight_matrix[:, []].sum(axis=1)
    weight_to_b = weight_matrix[:, idx_b].sum(axis=1) if idx_b else weight_matrix[:, []].sum(axis=1)

    best_delta = 0.0
    best_pair = None

    # Evaluate all cross pairs (i in A, j in B)
    for i in set_a:
        # Single flip gain for i: A -> B
        delta_i = float(weight_to_a[i] - weight_to_b[i])
        for j in set_b:
            # Single flip gain for j: B -> A
            delta_j = float(weight_to_b[j] - weight_to_a[j])

            # Correct double subtraction of the (i,j) cross-edge
            wij = float(weight_matrix[i, j])
            delta_pair = delta_i + delta_j + 2.0 * wij

            # Accept strictly positive improvements above threshold
            if delta_pair > min_gain_threshold:
                if early_stop_first_improvement:
                    return SwapOperator([i, j]), {}
                if delta_pair > best_delta:
                    best_delta = delta_pair
                    best_pair = (i, j)

    # Return the globally best improving pair if found
    if best_pair is not None:
        return SwapOperator([best_pair[0], best_pair[1]]), {}

    # No strictly improving pair found
    return None, {}