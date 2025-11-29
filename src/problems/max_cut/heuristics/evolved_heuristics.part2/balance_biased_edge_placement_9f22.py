from src.problems.max_cut.components import *
import numpy as np

def balance_biased_edge_placement_9f22(problem_state: dict,
                                       algorithm_data: dict,
                                       gamma: float = 0.1,
                                       degree_power: float = 1.0,
                                       pair_scan_limit: int = 0,
                                       matrix_mode_max_m: int = 1500,
                                       **kwargs):
    """
    Faster, cached and vectorized balance-biased pair insertion for MaxCut construction. For every unordered pair of unselected nodes {i, j}, evaluates both orientations (i→A, j→B) and (i→B, j→A) using a score that combines immediate cut gain and a balance-driven bonus. The bonus prefers sending the higher-degree node to the currently smaller side, controlled by gamma and degree_power. When pair_scan_limit <= 0 and the number of unselected nodes is moderate, all pairs are evaluated via full-matrix vectorization; otherwise, pairs are scanned in lexicographic chunks with vectorized row operations. Tie-breaking is deterministic, preferring the AB orientation on equal scores and preserving lexicographic order. The function maintains and advances caches (per-node sums to current sets and degree^power) in algorithm_data to accelerate subsequent calls, automatically refreshing them when the partition changes externally.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): Edge weight matrix W of shape (n, n). Treated as undirected; values are used as provided.
            - "current_solution" (Solution): Current partition with attributes set_a and set_b.
            - "unselected_nodes" (set[int]): Nodes not yet assigned to either set.
        algorithm_data (dict): The algorithm dictionary used for caching and will be updated. Internally, caches are stored under the key "bbep9f21_cache":
            - "deg_pow" (np.ndarray): Precomputed degree^degree_power for all nodes.
            - "sum_to_A" (np.ndarray): For each node v, sum of weights W[v, a] over a ∈ set A.
            - "sum_to_B" (np.ndarray): For each node v, sum of weights W[v, b] over b ∈ set B.
            - "set_a_snapshot" (set[int]), "set_b_snapshot" (set[int]): Snapshots of the current partition used to detect external changes and refresh caches.
            - "n" (int), "degree_power" (float): Metadata for cache validity.
        gamma (float): Trade-off weight multiplying the balance bonus. Larger values emphasize sending higher-degree nodes to the smaller side. Default is 0.1.
        degree_power (float): Exponent applied to node weighted degrees in the bonus. >1 strengthens preference for high-degree nodes; <1 dampens it. Default is 1.0.
        pair_scan_limit (int): Maximum number of unordered pairs evaluated (in lexicographic order). If <= 0, evaluates all available pairs (subject to matrix_mode_max_m). Default is 0.
        matrix_mode_max_m (int): Threshold on the number of unselected nodes for using full-matrix vectorization. If |U| > matrix_mode_max_m, falls back to lexicographic chunked evaluation to limit memory usage. Default is 1500.

    Returns:
        InsertEdgeOperator: Operator that inserts the selected pair with orientation i→A, j→B or i→B, j→A, maximizing the scored objective. If fewer than two unselected nodes exist or no viable pair is found, returns None.
        dict: Updated algorithm_data containing caches under "bbep9f21_cache" to speed up subsequent calls (caches are advanced to reflect the returned operator and auto-refresh if the partition changes externally).
    """
    W = problem_state.get("weight_matrix", None)
    current_solution = problem_state.get("current_solution", None)
    unselected_nodes = problem_state.get("unselected_nodes", None)

    if W is None or current_solution is None or unselected_nodes is None:
        return None, algorithm_data

    W = np.asarray(W)
    if W.ndim != 2:
        return None, algorithm_data
    n = W.shape[0]
    if len(unselected_nodes) < 2:
        return None, algorithm_data

    set_a = current_solution.set_a
    set_b = current_solution.set_b

    # --- Caches ---
    cache = algorithm_data.get('bbep9f21_cache', {})
    # Degree^power cache (independent of solution partition)
    deg_pow = cache.get('deg_pow', None)
    if deg_pow is None or cache.get('n') != n or cache.get('degree_power') != degree_power:
        deg = W.sum(axis=1).astype(np.float64)
        deg_pow = deg ** float(degree_power)
        cache['deg_pow'] = deg_pow
        cache['n'] = n
        cache['degree_power'] = float(degree_power)

    # sum_to_A / sum_to_B cache (depends on current partition)
    set_a_snapshot = cache.get('set_a_snapshot')
    set_b_snapshot = cache.get('set_b_snapshot')
    sum_to_A = cache.get('sum_to_A', None)
    sum_to_B = cache.get('sum_to_B', None)
    if (sum_to_A is None or sum_to_B is None or
        set_a_snapshot != set_a or set_b_snapshot != set_b):
        A_list = list(set_a)
        B_list = list(set_b)
        sum_to_A = W[:, A_list].sum(axis=1).astype(np.float64) if A_list else np.zeros(n, dtype=np.float64)
        sum_to_B = W[:, B_list].sum(axis=1).astype(np.float64) if B_list else np.zeros(n, dtype=np.float64)
        cache['sum_to_A'] = sum_to_A
        cache['sum_to_B'] = sum_to_B
        cache['set_a_snapshot'] = set(set_a)
        cache['set_b_snapshot'] = set(set_b)

    # Smaller side
    if len(set_a) < len(set_b):
        smaller_side = 'A'
    elif len(set_a) > len(set_b):
        smaller_side = 'B'
    else:
        smaller_side = None

    U = np.fromiter(sorted(unselected_nodes), dtype=np.int64)
    m = U.size
    if m < 2:
        algorithm_data['bbep9f21_cache'] = cache
        return None, algorithm_data

    # Helper references
    sA = sum_to_A[U]
    sB = sum_to_B[U]
    dU = deg_pow[U]

    best_score = -np.inf
    best_pair = None
    best_orientation = None  # 'AB' or 'BA'

    if pair_scan_limit <= 0 and m <= matrix_mode_max_m:
        # Full-matrix vectorization over all unordered pairs i<j
        W_sub = W[np.ix_(U, U)].astype(np.float64)

        score_ab = sB[:, None] + sA[None, :] + W_sub
        score_ba = sA[:, None] + sB[None, :] + W_sub

        if smaller_side is None:
            pass
        elif smaller_side == 'A':
            score_ab += gamma * dU[:, None]       # i -> A
            score_ba += gamma * dU[None, :]       # j -> A
        else:  # 'B'
            score_ab += gamma * dU[None, :]       # j -> B
            score_ba += gamma * dU[:, None]       # i -> B

        ab_pref = (score_ab >= score_ba)          # tie -> AB
        max_scores = np.where(ab_pref, score_ab, score_ba)

        # Mask lower triangle and diag
        mask = np.triu(np.ones_like(W_sub, dtype=bool), k=1)
        max_scores = np.where(mask, max_scores, -np.inf)

        flat_idx = int(max_scores.argmax())
        best_score = float(max_scores.ravel()[flat_idx])
        if not np.isfinite(best_score):
            algorithm_data['bbep9f21_cache'] = cache
            return None, algorithm_data

        i_idx, j_idx = np.unravel_index(flat_idx, max_scores.shape)
        best_orientation = 'AB' if bool(ab_pref[i_idx, j_idx]) else 'BA'
        best_pair = (int(U[i_idx]), int(U[j_idx]))
    else:
        # Lexicographic chunked evaluation honoring pair_scan_limit
        evaluated_pairs = 0
        for i_idx in range(m - 1):
            rem = (pair_scan_limit if pair_scan_limit > 0 else np.inf) - evaluated_pairs
            if rem <= 0:
                break
            j_count = int(min(rem, m - i_idx - 1))
            if j_count <= 0:
                continue

            i_node = U[i_idx]
            j_nodes = U[i_idx + 1 : i_idx + 1 + j_count]

            w_row = W[i_node, j_nodes].astype(np.float64)

            gain_ab = sB[i_idx] + sA[i_idx + 1 : i_idx + 1 + j_count] + w_row
            gain_ba = sA[i_idx] + sB[i_idx + 1 : i_idx + 1 + j_count] + w_row

            if smaller_side is None:
                score_ab = gain_ab
                score_ba = gain_ba
            elif smaller_side == 'A':
                score_ab = gain_ab + gamma * dU[i_idx]                               # i -> A
                score_ba = gain_ba + gamma * dU[i_idx + 1 : i_idx + 1 + j_count]     # j -> A
            else:  # 'B'
                score_ab = gain_ab + gamma * dU[i_idx + 1 : i_idx + 1 + j_count]     # j -> B
                score_ba = gain_ba + gamma * dU[i_idx]                               # i -> B

            ab_pref = (score_ab >= score_ba)  # tie -> AB
            max_scores = np.where(ab_pref, score_ab, score_ba)

            # Best in this chunk; np.argmax returns first index on ties -> preserves lex order within chunk
            chunk_idx = int(max_scores.argmax())
            chunk_best = float(max_scores[chunk_idx])

            if chunk_best > best_score:
                best_score = chunk_best
                best_orientation = 'AB' if bool(ab_pref[chunk_idx]) else 'BA'
                best_pair = (int(i_node), int(j_nodes[chunk_idx]))

            evaluated_pairs += j_count

    if best_pair is None:
        algorithm_data['bbep9f21_cache'] = cache
        return None, algorithm_data

    i, j = best_pair
    if best_orientation == 'AB':
        op = InsertEdgeOperator(node_1=i, node_2=j)
        # Advance caches to reflect applying op (so next call is fast)
        cache['sum_to_A'] = sum_to_A + W[:, i]
        cache['sum_to_B'] = sum_to_B + W[:, j]
        new_set_a = set(set_a); new_set_b = set(set_b)
        new_set_a.add(i); new_set_b.add(j)
        cache['set_a_snapshot'] = new_set_a
        cache['set_b_snapshot'] = new_set_b
    else:
        op = InsertEdgeOperator(node_1=j, node_2=i)
        cache['sum_to_A'] = sum_to_A + W[:, j]
        cache['sum_to_B'] = sum_to_B + W[:, i]
        new_set_a = set(set_a); new_set_b = set(set_b)
        new_set_a.add(j); new_set_b.add(i)
        cache['set_a_snapshot'] = new_set_a
        cache['set_b_snapshot'] = new_set_b

    algorithm_data['bbep9f21_cache'] = cache
    return op, algorithm_data