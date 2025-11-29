from src.problems.max_cut.components import *

from typing import Optional, Tuple

def degree_product_edge_chooser_97c4(
    problem_state: dict,
    algorithm_data: dict,
    top_k: int = 0,
    use_abs: bool = False,
    ignore_diagonal: bool = True,
    **kwargs
) -> Tuple[Optional["BaseOperator"], dict]:
    """
    Degree-product oriented edge insertion with fast extremal pairing and vectorized local cut gains.

    This constructive heuristic selects two currently unassigned vertices by maximizing the product of their (possibly signed) weighted degrees. It avoids scanning all O(|U|^2) pairs by keeping only the two largest positive degrees and two most negative degrees, which suffices to maximize the pairwise product among real numbers. The chosen pair is then oriented into opposite sets (A/B) based on which orientation yields the larger immediate increase in the cut value relative to the existing partition, computed via vectorized sums to sets A and B. If only one unassigned vertex remains (or top_k restriction leaves fewer than two candidates), it inserts that single vertex into the side with higher immediate gain.

    Complexity:
    - Degree computation: O(n) once; cached across calls in algorithm_data for fixed use_abs/ignore_diagonal.
    - Candidate restriction: O(|U|) via argpartition when top_k > 1.
    - Pair selection: O(k) (k = number of candidates after restriction).
    - Orientation gain: O(|A| + |B|) to compute sums once; O(1) per evaluated node.

    Args:
        problem_state (dict): Requires
            - "weight_matrix" (numpy.ndarray): Symmetric adjacency/weight matrix (n x n).
            - "current_solution" (Solution): Current partition with set_a (set[int]) and set_b (set[int]).
            - "unselected_nodes" (set[int]): Nodes not yet placed in either set.
        algorithm_data (dict): Will be updated with a cached degree vector keyed by (use_abs, ignore_diagonal, node_num).
        top_k (int): If > 1, restrict candidate nodes to the top_k unselected nodes by weighted degree (descending).
        use_abs (bool): If True, degrees = sum_j |w(i,j)|; otherwise degrees = sum_j w(i,j). The diagonal term is treated according to ignore_diagonal.
        ignore_diagonal (bool): If True, exclude w(i,i) from degree computation.

    Returns:
        (Optional[BaseOperator], dict):
            - InsertEdgeOperator(node_1→A, node_2→B) with orientation maximizing immediate cut gain.
            - Boundary: If fewer than two candidates remain, returns InsertNodeOperator for the single node with better immediate gain.
            - If no move is possible (no candidates), returns (None, algorithm_data).

    Notes:
        - Signed degrees: When use_abs=False, degrees can be negative. The maximum product over pairs is achieved by either the two largest positives or the two most negative degrees. If neither exists, the best available product may be zero (if a zero exists) or a negative product (largest positive with largest negative closest to zero).
        - Caching: The per-node degree vector is independent of the current partition and cached in algorithm_data["degree_cache"] keyed by (use_abs, ignore_diagonal, node_num).
        - Orientation: Immediate cut gain uses vectorized sums to A and B to avoid per-node Python loops.
    """
    import numpy as np

    W = problem_state["weight_matrix"]
    sol = problem_state["current_solution"]
    U = problem_state["unselected_nodes"]

    # No constructive move possible
    if not U:
        return None, algorithm_data

    A = sol.set_a
    B = sol.set_b
    n = W.shape[0]

    # Cache degrees across calls
    deg_cache = algorithm_data.get("degree_cache")
    if (
        deg_cache is None
        or deg_cache.get("use_abs") != use_abs
        or deg_cache.get("ignore_diagonal") != ignore_diagonal
        or deg_cache.get("node_num") != n
    ):
        if use_abs:
            degrees = np.sum(np.abs(W), axis=1)
            if ignore_diagonal:
                degrees = degrees - np.abs(np.diag(W))
        else:
            degrees = np.sum(W, axis=1)
            if ignore_diagonal:
                degrees = degrees - np.diag(W)
        algorithm_data["degree_cache"] = {
            "degrees": degrees,
            "use_abs": use_abs,
            "ignore_diagonal": ignore_diagonal,
            "node_num": n,
        }
    else:
        degrees = deg_cache["degrees"]

    # Vectorized sums to current sets for orientation
    if A:
        A_idx = np.fromiter(A, dtype=int)
        sum_to_A = W[:, A_idx].sum(axis=1)
    else:
        sum_to_A = np.zeros(n, dtype=W.dtype)

    if B:
        B_idx = np.fromiter(B, dtype=int)
        sum_to_B = W[:, B_idx].sum(axis=1)
    else:
        sum_to_B = np.zeros(n, dtype=W.dtype)

    U_idx = np.fromiter(U, dtype=int)

    # Single-node boundary case
    if U_idx.size == 1:
        lone = int(U_idx[0])
        gainA = float(sum_to_B[lone])  # put lone in A, interacts with B
        gainB = float(sum_to_A[lone])  # put lone in B, interacts with A
        target_set = "A" if gainA >= gainB else "B"
        return InsertNodeOperator(node=lone, target_set=target_set), algorithm_data

    # Optional restriction to top_k by descending degree
    if isinstance(top_k, int) and top_k > 1 and U_idx.size > top_k:
        deg_u = degrees[U_idx]
        part = np.argpartition(deg_u, -top_k)[-top_k:]
        cand_idx = U_idx[part]
        deg_cand = deg_u[part]
    else:
        cand_idx = U_idx
        deg_cand = degrees[U_idx]

    # Boundary after restriction
    if cand_idx.size < 2:
        best_node = int(cand_idx[0])
        gainA = float(sum_to_B[best_node])
        gainB = float(sum_to_A[best_node])
        target_set = "A" if gainA >= gainB else "B"
        return InsertNodeOperator(node=best_node, target_set=target_set), algorithm_data

    # O(k) extremal pair selection
    max1 = -np.inf
    max2 = -np.inf
    m1 = -1
    m2 = -1
    min1 = np.inf
    min2 = np.inf
    n1 = -1
    n2 = -1
    zero_positions = []
    # Track largest negative by value (closest to zero) for pos-neg fallback
    best_neg_val = -np.inf  # will store the largest negative (e.g., -0.1 > -1.0)
    best_neg_idx = -1

    for local_idx in range(deg_cand.size):
        d = float(deg_cand[local_idx])
        if d > 0:
            if d > max1:
                max2, m2 = max1, m1
                max1, m1 = d, local_idx
            elif d > max2:
                max2, m2 = d, local_idx
        elif d < 0:
            # Two most negative (by value: smallest) for neg-neg
            if d < min1:
                min2, n2 = min1, n1
                min1, n1 = d, local_idx
            elif d < min2:
                min2, n2 = d, local_idx
            # Largest negative (closest to zero) for pos-neg fallback
            if d > best_neg_val:
                best_neg_val = d
                best_neg_idx = local_idx
        else:
            zero_positions.append(local_idx)

    best_pair_local = None
    best_product = -np.inf

    # Two largest positives
    if m2 != -1:
        prod_pp = max1 * max2
        best_pair_local = (m1, m2)
        best_product = prod_pp

    # Two most negative (product is positive)
    if n2 != -1:
        prod_nn = min1 * min2
        if prod_nn > best_product:
            best_product = prod_nn
            best_pair_local = (n1, n2)

    # Fallbacks
    if best_pair_local is None:
        if zero_positions:
            # Pair a zero with the largest |degree| non-zero candidate to get product 0
            abs_deg = np.abs(deg_cand).copy()
            for zpos in zero_positions:
                abs_deg[zpos] = -1.0
            partner = int(np.argmax(abs_deg))
            zpos = zero_positions[0] if partner != -1 else (zero_positions[0] if len(zero_positions) >= 2 else -1)
            if zpos != -1 and partner != -1 and zpos != partner:
                best_pair_local = (zpos, partner)
                best_product = 0.0
            elif len(zero_positions) >= 2:
                best_pair_local = (zero_positions[0], zero_positions[1])
                best_product = 0.0
        else:
            # Only pos-neg pairs exist; choose the largest positive with the largest negative (closest to zero)
            if m1 != -1 and best_neg_idx != -1:
                best_pair_local = (m1, best_neg_idx)
                best_product = max1 * best_neg_val

    if best_pair_local is None:
        return None, algorithm_data

    i = int(cand_idx[best_pair_local[0]])
    j = int(cand_idx[best_pair_local[1]])

    # Orient the pair by immediate cut gain
    # Note: W[i, j] is added to both orientations (constant), included for completeness.
    delta_a_to_b = float(sum_to_B[i] + sum_to_A[j] + W[i, j])
    delta_b_to_a = float(sum_to_A[i] + sum_to_B[j] + W[i, j])

    operator = (
        InsertEdgeOperator(node_1=i, node_2=j)
        if delta_a_to_b >= delta_b_to_a
        else InsertEdgeOperator(node_1=j, node_2=i)
    )
    return operator, algorithm_data