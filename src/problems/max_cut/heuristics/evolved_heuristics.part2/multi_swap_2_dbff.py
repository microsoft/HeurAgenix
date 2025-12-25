from src.problems.max_cut.components import *
import numpy as np
import random

def multi_swap_2_dbff(problem_state: dict, algorithm_data: dict, min_delta: float=0.0, tie_break: str='random', eps: float=1e-12, **kwargs) -> tuple[SwapOperator, dict]:
    """Best-improvement pairwise swap (2-swap) local search with side-sum preprocessing and configurable acceptance/tie-breaking.
    Precomputes, for every vertex v, the sums of weights to current A and B (ΣA w(v,·), ΣB w(v,·)). For any pair (i ∈ A, j ∈ B), the simultaneous swap gain is:
        Δ(i,j) = (ΣA w(i,·) − ΣB w(i,·)) + (ΣB w(j,·) − ΣA w(j,·)) + 2·w(i,j).
    The last term corrects the double subtraction of edge (i,j) when combining single-flip gains, keeping the crossing edge’s contribution unchanged after the swap. The algorithm scans all pairs and selects the global best according to Δ, applying acceptance threshold min_delta and resolving ties via tie_break. This move helps escape 1-flip local optima (KL-style refinement). Time complexity: O(n|A| + n|B| + |A|·|B|); memory: O(n).

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): Symmetric adjacency/weight matrix W of shape (n, n).
            - "current_solution" (Solution): Current partition with sets `set_a` and `set_b`.
        algorithm_data (dict): Not used in this heuristic.
        min_delta (float): Minimum acceptable gain to perform the swap. Default is 0.0 (accept non-worsening moves). Set >0 to require strict improvement.
        tie_break (str): Tie-breaking rule among equal-gain pairs. One of {"max_edge", "first", "random"}.
                         "max_edge" prefers the pair with largest w(i,j); "first" keeps the earliest found; "random" samples uniformly among best candidates. Default is "random".
        eps (float): Numerical tolerance when comparing gains for tie detection. Default is 1e-12.

    Returns:
        SwapOperator: The operator that swaps the selected pair (i from A, j from B). Returns None if no admissible pair exists (e.g., one set empty or best gain < min_delta).
        dict: Empty dictionary as no algorithm data is updated.
    """
    current_solution = problem_state['current_solution']
    weight_matrix = problem_state['weight_matrix']

    set_a = current_solution.set_a
    set_b = current_solution.set_b

    # No feasible 2-swap if either side is empty.
    if not set_a or not set_b:
        return None, {}

    # Precompute side sums: for each vertex v, sum of weights to A and to B.
    idx_a = list(set_a)
    idx_b = list(set_b)
    weight_to_a = weight_matrix[:, idx_a].sum(axis=1)
    weight_to_b = weight_matrix[:, idx_b].sum(axis=1)

    best_increase = -np.inf
    best_candidates = []  # list of (i, j)

    for i in set_a:
        # Single-flip gain for moving i to B: g(i) = ΣA w(i,·) − ΣB w(i,·)
        gi = float(weight_to_a[i] - weight_to_b[i])
        for j in set_b:
            # Single-flip gain for moving j to A: g(j) = ΣB w(j,·) − ΣA w(j,·)
            gj = float(weight_to_b[j] - weight_to_a[j])

            # Combined simultaneous swap gain with correction for edge (i,j)
            delta = gi + gj + 2.0 * float(weight_matrix[i, j])

            if delta > best_increase + eps:
                best_increase = delta
                best_candidates = [(i, j)]
            elif abs(delta - best_increase) <= eps:
                best_candidates.append((i, j))

    # Apply acceptance threshold
    if best_increase < min_delta or not best_candidates:
        return None, {}

    # Resolve ties
    if tie_break == 'max_edge':
        # Prefer the pair with largest w(i,j)
        i_sel, j_sel = max(best_candidates, key=lambda p: float(weight_matrix[p[0], p[1]]))
    elif tie_break == 'random':
        i_sel, j_sel = random.choice(best_candidates)
    else:  # 'first'
        i_sel, j_sel = best_candidates[0]

    return SwapOperator([i_sel, j_sel]), {}