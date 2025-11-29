from src.problems.max_cut.components import *

def heavy_edge_matching_seed_edd5(
    problem_state: dict,
    algorithm_data: dict,
    pair_sample_ratio: float = 1.0,
    max_pair_evaluations: int = None,
    seed: int = None,
    **kwargs
) -> tuple[InsertEdgeOperator, dict]:
    """
    Greedy heavy-edge matching seed (HMS) for MaxCut with vectorized max-pair selection and optional partial row sampling.

    Key improvements:
    - Vectorized selection of the heaviest pair among unselected vertices via NumPy argmax on the induced submatrix.
    - Orientation decided with O(|A|+|B|) work for just the selected endpoints using phi(i)=sum_W(i,B)-sum_W(i,A),
      avoiding full O(n) arrays when only two values are needed.
    - Partial scanning is implemented by sampling rows (vertices) and taking the heaviest partner per sampled row,
      achieving approximately eval_limit pair evaluations in compiled code.

    Args:
        problem_state (dict):
            - "weight_matrix" (numpy.ndarray): adjacency matrix of edge weights.
            - "current_solution" (Solution): current partition; uses current_solution.set_a and current_solution.set_b.
            - "unselected_nodes" (set[int]): vertices not yet placed in either set.
        algorithm_data (dict): not used, but can carry caches if desired.
        pair_sample_ratio (float): fraction of total unordered pairs to evaluate. In (0,1]; 1.0 for full scan.
        max_pair_evaluations (int | None): optional hard cap; overrides ratio-derived budget when smaller.
        seed (int | None): RNG seed for row order/selection under partial scans.

    Returns:
        InsertEdgeOperator or InsertNodeOperator (single-node corner case), and a summary dict.
    """
    import math
    import numpy as np

    weight_matrix = np.asarray(problem_state["weight_matrix"])
    current_solution = problem_state["current_solution"]
    unselected_nodes = problem_state["unselected_nodes"]

    # No work if nothing to place.
    if not unselected_nodes:
        return None, {}

    # Single-node: place to best side w.r.t. current partition; O(|A|+|B|)
    if len(unselected_nodes) == 1:
        lone = next(iter(unselected_nodes))
        set_a_list = list(current_solution.set_a)
        set_b_list = list(current_solution.set_b)
        gain_to_a = float(weight_matrix[lone, set_b_list].sum()) if set_b_list else 0.0
        gain_to_b = float(weight_matrix[lone, set_a_list].sum()) if set_a_list else 0.0
        target_set = 'A' if gain_to_a >= gain_to_b else 'B'
        return InsertNodeOperator(node=lone, target_set=target_set), {
            "last_selected_pair": None,
            "evaluated_pairs": 0,
            "pair_sample_ratio": pair_sample_ratio,
            "max_pair_evaluations": max_pair_evaluations,
            "seed_used": seed
        }

    # Prepare unselected array and RNG
    unselected_arr = np.fromiter(unselected_nodes, dtype=np.int64)
    n_u = unselected_arr.shape[0]

    if seed is not None:
        rng = np.random.RandomState(seed)
        rng.shuffle(unselected_arr)
    else:
        unselected_arr.sort()

    total_pairs = n_u * (n_u - 1) // 2
    if pair_sample_ratio <= 0.0:
        eval_limit = 1
    else:
        eval_limit = int(math.ceil(pair_sample_ratio * total_pairs))
        eval_limit = max(1, min(eval_limit, total_pairs))
    if isinstance(max_pair_evaluations, int) and max_pair_evaluations > 0:
        eval_limit = min(eval_limit, max_pair_evaluations)

    # Decide between full vectorized scan and partial row sampling
    # Full scan: build submatrix and get global maximum in one pass (compiled).
    # Partial scan: sample rows; for each sampled row, take its max partner.
    if eval_limit >= total_pairs:
        # Full scan path
        sub = weight_matrix[np.ix_(unselected_arr, unselected_arr)].astype(float, copy=True)
        # Exclude self-pairs
        np.fill_diagonal(sub, -np.inf)
        # Row-wise maxima and then global maximum
        row_argmax = np.argmax(sub, axis=1)
        row_maxval = sub[np.arange(n_u), row_argmax]
        best_row = int(np.argmax(row_maxval))
        best_col = int(row_argmax[best_row])
        i = int(unselected_arr[best_row])
        j = int(unselected_arr[best_col])
        best_w = float(sub[best_row, best_col])
        evaluated = total_pairs
    else:
        # Partial row sampling: choose m_rows so that m_rows*(n_u-1) ≈ eval_limit
        m_rows = max(1, min(n_u, int(math.ceil(eval_limit / max(1, (n_u - 1))))))
        # Sample m_rows distinct row positions
        if seed is not None:
            rows_sel = rng.choice(n_u, size=m_rows, replace=False)
        else:
            # Deterministic: take the first m_rows in the sorted order
            rows_sel = np.arange(m_rows, dtype=np.int64)

        # Build a block of selected rows against all unselected columns
        block = weight_matrix[unselected_arr[rows_sel]][:, unselected_arr].astype(float, copy=True)
        # Mask diagonals for each selected row: diag column index equals the row's position in unselected_arr
        block[np.arange(m_rows), rows_sel] = -np.inf

        block_argmax = np.argmax(block, axis=1)
        block_maxval = block[np.arange(m_rows), block_argmax]
        best_block_row = int(np.argmax(block_maxval))
        best_row_pos = int(rows_sel[best_block_row])
        best_col_pos = int(block_argmax[best_block_row])

        i = int(unselected_arr[best_row_pos])
        j = int(unselected_arr[best_col_pos])
        best_w = float(block[best_block_row, best_col_pos])
        evaluated = m_rows * (n_u - 1)

    # Orientation via phi(i) - phi(j); only compute sums for i and j.
    set_a_list = list(current_solution.set_a)
    set_b_list = list(current_solution.set_b)
    sum_i_B = float(weight_matrix[i, set_b_list].sum()) if set_b_list else 0.0
    sum_i_A = float(weight_matrix[i, set_a_list].sum()) if set_a_list else 0.0
    sum_j_B = float(weight_matrix[j, set_b_list].sum()) if set_b_list else 0.0
    sum_j_A = float(weight_matrix[j, set_a_list].sum()) if set_a_list else 0.0

    phi_i = sum_i_B - sum_i_A
    phi_j = sum_j_B - sum_j_A

    if phi_i >= phi_j:
        operator = InsertEdgeOperator(node_1=i, node_2=j)  # i→A, j→B
    else:
        operator = InsertEdgeOperator(node_1=j, node_2=i)  # j→A, i→B

    return operator, {
        "last_selected_pair": (i, j),
        "evaluated_pairs": int(evaluated),
        "pair_sample_ratio": pair_sample_ratio,
        "max_pair_evaluations": max_pair_evaluations,
        "seed_used": seed
    }