from src.problems.max_cut.components import *
import random

def highest_delta_edge_9f66(problem_state: dict, algorithm_data: dict, sample_pairs_ratio: float=1.0, **kwargs) -> tuple[InsertEdgeOperator, dict]:
    """Greedy pairwise orientation with optional pair sampling. At each call, evaluates unordered pairs of currently unassigned vertices and both orientations (i→A, j→B vs i→B, j→A), selecting the orientation that maximizes immediate cut gain computed from precomputed affiliation sums to the current sets. For efficiency, an adjustable fraction of pairs can be sampled uniformly to reduce O(|U|^2) scanning while retaining strong improvement potential. If exactly one vertex remains unselected, it is paired with the best counterpart from the already selected side to implement the best single-vertex insertion via an InsertEdgeOperator; if both sets are empty, falls back to a single-vertex insertion. Weighted and asymmetric matrices are supported.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): Weighted adjacency matrix; can be asymmetric.
            - "current_solution" (Solution): Current partition with sets A and B.
            - "unselected_nodes" (set[int]): Vertices not yet assigned to either set.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this heuristic.
        sample_pairs_ratio (float): Fraction of unordered pairs of unselected vertices to evaluate (0 < sample_pairs_ratio <= 1). 
            - If >= 1.0, all pairs are scanned (full greedy).
            - If < 1.0, a uniform random sample of pairs is scanned (diversification with reduced runtime).
            Default is 1.0.

    Returns:
        InsertEdgeOperator: Operator that inserts the chosen pair with the orientation maximizing immediate cut gain. 
            - If only one unselected vertex remains and a counterpart exists in the opposite set, returns an InsertEdgeOperator pairing the unselected vertex with that selected counterpart to effect the insertion.
            - If both sets are empty and exactly one vertex remains, returns an InsertNodeOperator as a safe fallback.
        dict: Empty dictionary as no algorithm data is updated.
    """
    weight_matrix = problem_state["weight_matrix"]
    current_solution: Solution = problem_state["current_solution"]
    unselected_nodes: set[int] = problem_state["unselected_nodes"]

    # If there are no unselected nodes left, no insertion is possible.
    if not unselected_nodes:
        return None, {}

    n = len(weight_matrix)

    # Precompute the sum of weights from each node to current sets A and B.
    delta_to_A = [sum(weight_matrix[i][other] for other in current_solution.set_a) for i in range(n)]
    delta_to_B = [sum(weight_matrix[i][other] for other in current_solution.set_b) for i in range(n)]

    unselected_list = list(unselected_nodes)

    # Handle the case of a single remaining unselected node by pairing with the best selected counterpart.
    if len(unselected_list) == 1:
        i = unselected_list[0]

        # Try pairing i→A with j in B
        best_j_B = None
        best_val_B = float('-inf')
        for j in current_solution.set_b:
            val = delta_to_A[j] + weight_matrix[i][j]  # sum_w(j,A) + w(i,j)
            if val > best_val_B:
                best_val_B = val
                best_j_B = j
        delta_i_to_A = delta_to_B[i] + (best_val_B if best_j_B is not None else float('-inf'))

        # Try pairing i→B with j in A
        best_j_A = None
        best_val_A = float('-inf')
        for j in current_solution.set_a:
            val = delta_to_B[j] + weight_matrix[i][j]  # sum_w(j,B) + w(i,j)
            if val > best_val_A:
                best_val_A = val
                best_j_A = j
        delta_i_to_B = delta_to_A[i] + (best_val_A if best_j_A is not None else float('-inf'))

        # Decide orientation based on the best available counterpart; if none exist, insert the lone node.
        if delta_i_to_A >= delta_i_to_B and best_j_B is not None:
            operator = InsertEdgeOperator(node_1=i, node_2=best_j_B)  # i→A, j→B
            return operator, {}
        elif best_j_A is not None:
            operator = InsertEdgeOperator(node_1=best_j_A, node_2=i)  # j→A, i→B
            return operator, {}
        else:
            # Both sets are empty; insert the single node into A as a safe fallback.
            operator = InsertNodeOperator(node=i, target_set='A')
            return operator, {}

    # Build all unordered pairs of unselected nodes.
    pairs = []
    for idx, u in enumerate(unselected_list):
        for v in unselected_list[idx + 1:]:
            pairs.append((u, v))

    # Sample a fraction of pairs if requested.
    ratio = max(0.0, min(1.0, sample_pairs_ratio))
    if ratio < 1.0:
        sample_size = max(1, int(round(ratio * len(pairs))))
        candidate_pairs = random.sample(pairs, sample_size)
    else:
        candidate_pairs = pairs

    best_delta = float('-inf')
    best_edge = None
    best_orientation = None  # 'i_to_A' or 'i_to_B'

    # Evaluate candidate pairs under both orientations.
    for (i, j) in candidate_pairs:
        delta_a_to_b = delta_to_B[i] + delta_to_A[j] + weight_matrix[i][j]  # i→A, j→B
        delta_b_to_a = delta_to_A[i] + delta_to_B[j] + weight_matrix[i][j]  # i→B, j→A

        if delta_a_to_b >= delta_b_to_a:
            delta = delta_a_to_b
            orientation = 'i_to_A'
        else:
            delta = delta_b_to_a
            orientation = 'i_to_B'

        if delta > best_delta:
            best_delta = delta
            best_edge = (i, j)
            best_orientation = orientation

    # If no edge is found (should not happen when |U|>=2), return None for safety.
    if best_edge is None:
        return None, {}

    i, j = best_edge
    if best_orientation == 'i_to_A':
        operator = InsertEdgeOperator(node_1=i, node_2=j)  # i→A, j→B
    else:
        operator = InsertEdgeOperator(node_1=j, node_2=i)  # j→A, i→B

    return operator, {}