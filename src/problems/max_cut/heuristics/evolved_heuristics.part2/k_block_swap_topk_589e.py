from src.problems.max_cut.components import *
import numpy as np

def k_block_swap_topk_589e(problem_state: dict, algorithm_data: dict, k: int = 1, require_net_positive: bool = False, min_net_gain: float = 0.0, **kwargs) -> tuple[SwapOperator, dict]:
    """Block flip of top-k improvement candidates from both sides (A and B) simultaneously with optional interaction-corrected acceptance.

    This heuristic computes the single-vertex flip gains for all currently assigned vertices and selects up to k nodes with strictly positive gains from set A and up to k from set B. It then flips all selected nodes at once via a single SwapOperator. Optionally, it evaluates the true net multi-flip gain by correcting for pairwise interactions among flipped nodes and only applies the move if the net gain exceeds a threshold. This accelerates local improvement by bundling several promising flips while guarding against overestimation from additive gains.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): Symmetric adjacency/weight matrix of the undirected graph (shape: [n, n]).
            - "current_solution" (Solution): The current partition, with disjoint sets current_solution.set_a and current_solution.set_b.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this heuristic.
        k (int): Maximum number of top positive-gain nodes to flip from each side (A and B). If k <= 0, no operation is performed. Default is 1.
        require_net_positive (bool): If True, computes interaction-corrected net gain of the multi-flip and applies the move only if net_gain > min_net_gain. Default is False.
        min_net_gain (float): Minimal net gain threshold used when require_net_positive is True. Default is 0.0.

    Returns:
        SwapOperator: The operator that flips the selected nodes (up to k from A and up to k from B) simultaneously; returns None if no eligible nodes or the net gain check fails.
        dict: Empty dictionary as no algorithm data is updated.

    """
    # Early exit if k is non-positive
    if k <= 0:
        return None, {}

    # Extract required state
    weight_matrix: np.ndarray = problem_state['weight_matrix']
    current_solution: Solution = problem_state['current_solution']
    set_a = current_solution.set_a
    set_b = current_solution.set_b

    # If no assigned nodes exist, nothing to flip
    if (not set_a) and (not set_b):
        return None, {}

    # Precompute sums of weights from each node to A and to B
    indices_a = list(set_a)
    indices_b = list(set_b)
    weight_to_a = weight_matrix[:, indices_a].sum(axis=1) if indices_a else np.zeros(weight_matrix.shape[0])
    weight_to_b = weight_matrix[:, indices_b].sum(axis=1) if indices_b else np.zeros(weight_matrix.shape[0])

    # Compute individual flip gains and keep only strictly positive candidates
    deltas_a = []
    for i in set_a:
        delta_i = float(weight_to_a[i] - weight_to_b[i])  # gain if i moves A -> B
        if delta_i > 0:
            deltas_a.append((i, delta_i))

    deltas_b = []
    for j in set_b:
        delta_j = float(weight_to_b[j] - weight_to_a[j])  # gain if j moves B -> A
        if delta_j > 0:
            deltas_b.append((j, delta_j))

    # If no positive-gain candidates exist on both sides, no move is possible
    if not deltas_a and not deltas_b:
        return None, {}

    # Select top-k by gain from each side
    deltas_a.sort(key=lambda x: x[1], reverse=True)
    deltas_b.sort(key=lambda x: x[1], reverse=True)
    selected_a = [node for node, _gain in deltas_a[:k]]
    selected_b = [node for node, _gain in deltas_b[:k]]

    # If both selections are empty, no move is possible
    if (not selected_a) and (not selected_b):
        return None, {}

    # Optional interaction-corrected net gain check
    if require_net_positive:
        # Sum of individual gains
        sum_individual = 0.0
        for i in selected_a:
            sum_individual += float(weight_to_a[i] - weight_to_b[i])
        for j in selected_b:
            sum_individual += float(weight_to_b[j] - weight_to_a[j])

        # Corrections: within-block pairs subtract 2*w; across-block pairs add 2*w
        correction_within_a = 0.0
        if len(selected_a) >= 2:
            for u_idx in range(len(selected_a)):
                u = selected_a[u_idx]
                for v_idx in range(u_idx + 1, len(selected_a)):
                    v = selected_a[v_idx]
                    correction_within_a += 2.0 * float(weight_matrix[u, v])

        correction_within_b = 0.0
        if len(selected_b) >= 2:
            for u_idx in range(len(selected_b)):
                u = selected_b[u_idx]
                for v_idx in range(u_idx + 1, len(selected_b)):
                    v = selected_b[v_idx]
                    correction_within_b += 2.0 * float(weight_matrix[u, v])

        correction_across = 0.0
        if selected_a and selected_b:
            for i in selected_a:
                for j in selected_b:
                    correction_across += 2.0 * float(weight_matrix[i, j])

        net_gain = sum_individual - correction_within_a - correction_within_b + correction_across

        if net_gain <= min_net_gain:
            return None, {}

    # Build and return the swap operator
    nodes_to_flip = selected_a + selected_b
    return SwapOperator(nodes_to_flip), {}