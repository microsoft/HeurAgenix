from src.problems.max_cut.components import *
import numpy as np
from typing import Optional

def multi_flip_threshold_fd21(problem_state: dict, algorithm_data: dict, tau: float = 0.0, max_batch_size: Optional[int] = None, epsilon: float = 0.0, **kwargs) -> tuple[SwapOperator, dict]:
    """Batch improvement via thresholded multi-node flips for undirected MaxCut.
    
    This heuristic evaluates, for each currently assigned vertex, the change in cut value if the vertex is flipped
    to the opposite set. All vertices with flip gain Δ ≥ tau + epsilon are flipped simultaneously using one SwapOperator.
    For a node i in A: Δ(i) = sum_w(i, A) - sum_w(i, B); for a node j in B: Δ(j) = sum_w(j, B) - sum_w(j, A).
    On undirected graphs (symmetric weights), this equals the exact single-flip cut improvement. Flipping multiple nodes
    at once accelerates improvement compared to sequential flips; interactions among flipped nodes are intentional.
    If no node meets the threshold, no operator is returned.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): Symmetric adjacency/weight matrix; shape (n, n).
            - "current_solution" (Solution): Current partition with disjoint sets set_a and set_b containing node indices.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this heuristic.
        tau (float): Minimum flip gain required to include a node in the batch. Default is 0.0 (flip non-worsening nodes).
        max_batch_size (int or None): Upper bound on the number of nodes flipped in one batch. If None or <= 0, all qualifying nodes are flipped. Default is None.
        epsilon (float): Numerical tolerance added to the threshold comparison (Δ ≥ tau + epsilon). Useful with floating weights. Default is 0.0.

    Returns:
        SwapOperator: The operator that swaps a batch of selected nodes to the opposite set to improve (or not worsen) the cut value. Returns None if no node qualifies.
        dict: Empty dictionary as no algorithm data is updated.
    """
    weight_matrix = problem_state["weight_matrix"]
    current_solution = problem_state["current_solution"]

    # Access sets and constrain indices to valid range to avoid out-of-bounds.
    n = weight_matrix.shape[0]
    set_a = {i for i in current_solution.set_a if 0 <= i < n}
    set_b = {j for j in current_solution.set_b if 0 <= j < n}

    # Precompute per-node sums to each side.
    idx_a = list(set_a)
    idx_b = list(set_b)
    weight_to_a = weight_matrix[:, idx_a].sum(axis=1) if len(idx_a) > 0 else np.zeros(n)
    weight_to_b = weight_matrix[:, idx_b].sum(axis=1) if len(idx_b) > 0 else np.zeros(n)

    # Collect candidates meeting the threshold, deterministic by sorted node order.
    candidates = []
    for i in sorted(set_a):
        delta_i = float(weight_to_a[i] - weight_to_b[i])
        if delta_i >= (tau + epsilon):
            candidates.append((i, delta_i))
    for j in sorted(set_b):
        delta_j = float(weight_to_b[j] - weight_to_a[j])
        if delta_j >= (tau + epsilon):
            candidates.append((j, delta_j))

    if not candidates:
        return None, {}

    # Apply optional batch cap: select top-k by delta (descending), tie-break by node id.
    candidates.sort(key=lambda x: (-x[1], x[0]))
    if max_batch_size is not None and max_batch_size > 0:
        selected_nodes = [node for node, _ in candidates[:max_batch_size]]
    else:
        selected_nodes = [node for node, _ in candidates]

    if not selected_nodes:
        return None, {}

    op = SwapOperator(nodes=selected_nodes)
    return op, {}