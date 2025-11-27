from src.problems.max_cut.components import *
import numpy as np

def balance_biased_edge_placement_9f21(problem_state: dict, algorithm_data: dict, gamma: float = 0.1, degree_power: float = 1.0, pair_scan_limit: int = 0, **kwargs) -> tuple[InsertEdgeOperator, dict]:
    """
    Balance-biased pair insertion for MaxCut construction. For every unordered pair of unselected nodes {i, j}, evaluates both orientations (i→A, j→B) and (i→B, j→A) using a score that combines immediate cut gain and a balance-driven bonus. The bonus prefers sending the higher-degree node to the currently smaller side, controlled by gamma and degree_power. Selects the orientation with the highest score and returns an InsertEdgeOperator for that pair. Deterministic evaluation order with an optional cap on the number of evaluated pairs via pair_scan_limit.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): Edge weight matrix W of shape (n, n). Treated as undirected; values are used as provided.
            - "current_solution" (Solution): Current partition with attributes set_a and set_b.
            - "unselected_nodes" (set[int]): Nodes not yet assigned to either set.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this heuristic.
        gamma (float): Trade-off weight multiplying the balance bonus. Larger values emphasize sending higher-degree nodes to the smaller side. Default is 0.1.
        degree_power (float): Exponent applied to node weighted degrees in the bonus. >1 strengthens preference for high-degree nodes; <1 dampens it. Default is 1.0.
        pair_scan_limit (int): Maximum number of unordered pairs evaluated (in lexicographic order). If <= 0, evaluates all available pairs. Default is 0.

    Returns:
        InsertEdgeOperator: Operator that inserts the selected pair with orientation i→A, j→B or i→B, j→A, maximizing the scored objective. If fewer than two unselected nodes exist or no viable pair is found, returns None.
        dict: Empty dictionary, as this heuristic does not update algorithm_data.
    """
    W = problem_state.get("weight_matrix", None)
    current_solution: Solution = problem_state.get("current_solution", None)
    unselected_nodes = problem_state.get("unselected_nodes", None)

    if W is None or current_solution is None or unselected_nodes is None:
        return None, {}

    W = np.asarray(W)
    if W.ndim != 2:
        return None, {}
    n = W.shape[0]

    # Need at least two unselected nodes to place an edge-oriented pair
    if not unselected_nodes or len(unselected_nodes) < 2:
        return None, {}

    set_a = current_solution.set_a
    set_b = current_solution.set_b

    A_list = list(set_a)
    B_list = list(set_b)

    sum_to_A = W[:, A_list].sum(axis=1) if len(A_list) > 0 else np.zeros(n, dtype=float)
    sum_to_B = W[:, B_list].sum(axis=1) if len(B_list) > 0 else np.zeros(n, dtype=float)

    deg = W.sum(axis=1) if (W.shape[0] == W.shape[1]) else np.zeros(n, dtype=float)

    if len(set_a) < len(set_b):
        smaller_side = 'A'
    elif len(set_a) > len(set_b):
        smaller_side = 'B'
    else:
        smaller_side = None

    U = sorted(unselected_nodes)
    best_score = -float('inf')
    best_pair = None
    best_orientation = None  # 'AB' for i→A, j→B; 'BA' for i→B, j→A

    evaluated_pairs = 0
    max_pairs = pair_scan_limit if pair_scan_limit > 0 else float('inf')

    for idx_i in range(len(U)):
        i = U[idx_i]
        if not (0 <= i < n):
            continue
        for idx_j in range(idx_i + 1, len(U)):
            j = U[idx_j]
            if not (0 <= j < n):
                continue

            if evaluated_pairs >= max_pairs:
                break
            evaluated_pairs += 1

            wij = float(W[i, j])

            gain_ab = float(sum_to_B[i]) + float(sum_to_A[j]) + wij
            gain_ba = float(sum_to_A[i]) + float(sum_to_B[j]) + wij

            if smaller_side is None:
                bonus_ab = 0.0
                bonus_ba = 0.0
            else:
                di = float(deg[i]) ** degree_power
                dj = float(deg[j]) ** degree_power
                if smaller_side == 'A':
                    bonus_ab = di   # i goes to A
                    bonus_ba = dj   # j goes to A
                else:  # smaller_side == 'B'
                    bonus_ab = dj   # j goes to B
                    bonus_ba = di   # i goes to B

            score_ab = gain_ab + gamma * bonus_ab
            score_ba = gain_ba + gamma * bonus_ba

            if score_ab >= score_ba:
                if score_ab > best_score:
                    best_score = score_ab
                    best_pair = (i, j)
                    best_orientation = 'AB'
            else:
                if score_ba > best_score:
                    best_score = score_ba
                    best_pair = (i, j)
                    best_orientation = 'BA'
        if evaluated_pairs >= max_pairs:
            break

    if best_pair is None or best_orientation is None:
        return None, {}

    i, j = best_pair
    if best_orientation == 'AB':
        op = InsertEdgeOperator(node_1=i, node_2=j)
    else:
        op = InsertEdgeOperator(node_1=j, node_2=i)

    return op, {}