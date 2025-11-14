from src.problems.max_cut.components import *
import numpy as np

def spectral_seed_fiedler_51e0(
    problem_state: dict,
    algorithm_data: dict,
    matrix_choice: str = "laplacian",
    laplacian_type: str = "unnormalized",
    symmetrize: bool = True,
    max_dense_size: int = 1024,
    zero_eig_tol: float = 1e-10,
    power_iter_max_iter: int = 200,
    power_iter_tol: float = 1e-8,
    use_abs_selection: bool = True,
    eps_abs_score_thresh: float = 1e-12,
    tie_break: str = "balance",
    **kwargs
) -> tuple[InsertNodeOperator, dict]:
    """
    Spectral constructive seeding with polarity alignment. Inserts exactly one unselected node into set A or B
    using a spectral score vector v: positive → A, negative → B. Prefers the Fiedler (Laplacian) or leading
    adjacency eigenvector, but also supports a direct per-node score input via problem_state["weighted_degree_distribution"].
    Robust fallbacks ensure an operator is returned whenever unselected nodes exist:
    - If spectral signal is weak or unavailable, a degree-centered proxy is used.
    - If no matrix/score is usable, a balance-based tie-break inserts an arbitrary unselected node into the smaller set.

    Uniqueness:
    - Polarity alignment to current partition stabilizes eigenvector sign ambiguity so “positive” consistently maps to set A.
    - Direct-score fast path: if a per-node score is provided, it is used as v without eigendecomposition.
    - Weak-signal and invalid-input resilience: gracefully falls back to degree proxy or balanced insertion.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): Current partition with sets A and B; used for polarity alignment and tie-breaks.
            - "unselected_nodes" (set[int]): Nodes not yet assigned to either set; one will be inserted.
            - "weight_matrix" (numpy.ndarray): Square adjacency/weight matrix n×n; required unless a direct score is provided.
            - (Optional) "weighted_degree_distribution" (array-like of length n): Precomputed node scores; used directly as v if present.
            - (Optional) "node_num" (int): Total number of vertices; Not used in this heuristic.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this heuristic.

        Hyper-parameters:
            - matrix_choice (str): Spectral basis when computing from weight_matrix. Default "laplacian".
                Choices:
                    "laplacian": use Laplacian’s Fiedler vector (second-smallest eigenvector).
                    "adjacency": use the leading eigenvector of the symmetrized adjacency.
            - laplacian_type (str): Laplacian variant if matrix_choice="laplacian". Default "unnormalized".
                Choices:
                    "unnormalized": L = D − W_sym.
                    "normalized":   L = I − D^{-1/2} W_sym D^{-1/2}.
            - symmetrize (bool): Symmetrize W as W_sym = 0.5*(W + W^T). Default True.
            - max_dense_size (int): Use dense eigendecomposition if n ≤ max_dense_size; otherwise power iteration. Default 1024.
            - zero_eig_tol (float): Tolerance for near-zero Laplacian eigenvalues to detect multiplicity. Default 1e-10.
            - power_iter_max_iter (int): Max iterations for power iteration. Default 200.
            - power_iter_tol (float): Convergence tolerance for power iteration. Default 1e-8.
            - use_abs_selection (bool): Select unselected node by maximum absolute spectral score (|v|). Default True.
            - eps_abs_score_thresh (float): If max |score| among unselected ≤ threshold, switch to degree-based proxy. Default 1e-12.
            - tie_break (str): Orientation when chosen score equals zero or no signal. Default "balance".
                Choices:
                    "balance": insert into smaller set (ties → A).
                    "A": force insertion into A.
                    "B": force insertion into B.

    Returns:
        InsertNodeOperator: Operator inserting one unselected node into A or B, chosen by spectral score or robust fallback.
        dict: Empty dictionary (no algorithm data updates). Returns (None, {}) only if there are no unselected nodes.
    """
    # Extract required inputs
    current_solution: Solution = problem_state.get("current_solution", None)
    unselected_nodes: set[int] = problem_state.get("unselected_nodes", set())

    # If there are no unselected nodes, return None
    if current_solution is None or not isinstance(unselected_nodes, set) or len(unselected_nodes) == 0:
        return None, {}

    W = problem_state.get("weight_matrix", None)
    v_direct = problem_state.get("weighted_degree_distribution", None)

    # Helper: balanced insertion fallback
    def balanced_insertion_fallback() -> InsertNodeOperator:
        node = next(iter(unselected_nodes))
        set_a_size = len(current_solution.set_a) if current_solution.set_a is not None else 0
        set_b_size = len(current_solution.set_b) if current_solution.set_b is not None else 0
        if tie_break == "A":
            target = "A"
        elif tie_break == "B":
            target = "B"
        else:
            target = "A" if set_a_size <= set_b_size else "B"
        return InsertNodeOperator(node=node, target_set=target)

    # Helper: power iteration for symmetric matrices
    def power_iteration_leading_eigenvector(A: np.ndarray, max_iter: int, tol: float) -> np.ndarray:
        n_local = A.shape[0]
        rng = np.random.default_rng()
        x = rng.standard_normal(n_local)
        nx = np.linalg.norm(x)
        if nx <= 0 or not np.isfinite(nx):
            x = np.ones(n_local, dtype=float)
            nx = np.linalg.norm(x)
        x = x / (nx if nx > 0 else 1.0)
        for _ in range(max_iter):
            y = A @ x
            ny = np.linalg.norm(y)
            if ny <= 0 or not np.isfinite(ny):
                return x
            y = y / ny
            if np.linalg.norm(y - x) <= tol:
                return y
            x = y
        return x

    # Determine spectral vector v and problem size n
    v = None
    n = None

    # Fast path: direct per-node score
    if v_direct is not None:
        v_arr = np.asarray(v_direct, dtype=float).reshape(-1)
        n = v_arr.shape[0]
        v = v_arr.copy()
    else:
        # Need a valid weight matrix to compute spectral scores; otherwise fallback to balanced insertion
        if W is None:
            operator = balanced_insertion_fallback()
            return operator, {}
        W = np.asarray(W)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            operator = balanced_insertion_fallback()
            return operator, {}
        n = W.shape[0]
        W_sym = 0.5 * (W + W.T) if symmetrize else W

        try:
            if matrix_choice == "laplacian":
                deg = W_sym.sum(axis=1)
                if laplacian_type == "unnormalized":
                    L = np.diag(deg) - W_sym
                else:
                    inv_sqrt_deg = np.zeros_like(deg, dtype=float)
                    mask = deg > 0
                    inv_sqrt_deg[mask] = 1.0 / np.sqrt(deg[mask])
                    D_inv_sqrt = np.diag(inv_sqrt_deg)
                    L = np.eye(n, dtype=float) - (D_inv_sqrt @ W_sym @ D_inv_sqrt)

                if n <= max_dense_size:
                    evals, evecs = np.linalg.eigh(L)
                    zero_count = int(np.sum(evals <= zero_eig_tol))
                    fiedler_index = min(max(zero_count, 1), n - 1)  # ensure at least index 1
                    v = evecs[:, fiedler_index]
                else:
                    # Use adjacency power iteration as scalable surrogate
                    v = power_iteration_leading_eigenvector(W_sym, power_iter_max_iter, power_iter_tol)
            else:
                # Adjacency basis
                if n <= max_dense_size:
                    evals, evecs = np.linalg.eigh(W_sym)
                    v = evecs[:, -1]
                else:
                    v = power_iteration_leading_eigenvector(W_sym, power_iter_max_iter, power_iter_tol)
        except Exception:
            # Numerical fallback: adjacency power iteration
            v = power_iteration_leading_eigenvector(W_sym, power_iter_max_iter, power_iter_tol)

    # If spectral vector invalid, fallback to balanced insertion
    if v is None or n is None or v.shape[0] != n:
        operator = balanced_insertion_fallback()
        return operator, {}

    # Polarity alignment to stabilize sign ambiguity
    A_idx = [i for i in current_solution.set_a if 0 <= i < n]
    B_idx = [i for i in current_solution.set_b if 0 <= i < n]
    if len(A_idx) > 0 and len(B_idx) > 0:
        mean_A = float(np.mean(v[A_idx])) if len(A_idx) > 0 else 0.0
        mean_B = float(np.mean(v[B_idx])) if len(B_idx) > 0 else 0.0
        if np.isfinite(mean_A) and np.isfinite(mean_B) and (mean_A < mean_B):
            v = -v
    else:
        if len(A_idx) > 0:
            mean_A = float(np.mean(v[A_idx]))
            if np.isfinite(mean_A) and mean_A < 0.0:
                v = -v
        elif len(B_idx) > 0:
            mean_B = float(np.mean(v[B_idx]))
            if np.isfinite(mean_B) and mean_B > 0.0:
                v = -v

    # Spectral scores for unselected nodes; sanitize NaN/Inf
    scores = {}
    for node in unselected_nodes:
        if 0 <= node < n:
            val = v[node]
            scores[node] = val if np.isfinite(val) else 0.0

    # If no unselected nodes map into [0, n), fallback to balanced insertion
    if len(scores) == 0:
        operator = balanced_insertion_fallback()
        return operator, {}

    max_abs_score = max(abs(val) for val in scores.values())

    # Select node using weak-signal proxy or spectral policy
    chosen_node = None
    chosen_value = 0.0
    if max_abs_score <= eps_abs_score_thresh:
        if W is None:
            chosen_node = next(iter(unselected_nodes))
            chosen_value = 0.0
        else:
            W_use = 0.5 * (W + W.T) if symmetrize else W
            deg = W_use.sum(axis=1)
            mean_deg = float(np.mean(deg)) if deg.size > 0 else 0.0
            centered_deg = deg - mean_deg
            candidates = [(node, abs(centered_deg[node])) for node in unselected_nodes if 0 <= node < n]
            if len(candidates) == 0:
                operator = balanced_insertion_fallback()
                return operator, {}
            chosen_node, _ = max(candidates, key=lambda x: x[1])
            chosen_value = centered_deg[chosen_node]
    else:
        if use_abs_selection:
            chosen_node, _ = max(((node, abs(scores[node])) for node in scores.keys()), key=lambda x: x[1])
            chosen_value = scores[chosen_node]
        else:
            pos_items = [(node, scores[node]) for node in scores.keys() if scores[node] > 0]
            if len(pos_items) > 0:
                chosen_node, chosen_value = max(pos_items, key=lambda x: x[1])
            else:
                neg_items = [(node, scores[node]) for node in scores.keys() if scores[node] < 0]
                if len(neg_items) == 0:
                    operator = balanced_insertion_fallback()
                    return operator, {}
                else:
                    chosen_node, chosen_value = min(neg_items, key=lambda x: x[1])  # most negative

    # Determine target set based on sign or tie-break policy
    if chosen_value > 0:
        target_set = "A"
    elif chosen_value < 0:
        target_set = "B"
    else:
        if tie_break == "A":
            target_set = "A"
        elif tie_break == "B":
            target_set = "B"
        else:
            set_a_size = len(current_solution.set_a) if current_solution.set_a is not None else 0
            set_b_size = len(current_solution.set_b) if current_solution.set_b is not None else 0
            target_set = "A" if set_a_size <= set_b_size else "B"

    # Final validity guard
    if (chosen_node in current_solution.set_a) or (chosen_node in current_solution.set_b):
        operator = balanced_insertion_fallback()
        return operator, {}

    return InsertNodeOperator(node=chosen_node, target_set=target_set), {}