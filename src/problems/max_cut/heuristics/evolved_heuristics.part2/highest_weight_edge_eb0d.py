from src.problems.max_cut.components import *

from typing import Optional, Tuple
def highest_weight_edge_eb0d(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertEdgeOperator, dict]:
    """
    Vectorized greedy constructive seeding by globally heaviest unselected edge. Uses NumPy to select the edge (i,j) with maximum weight among unselected node pairs via a single argmax over the |U|×|U| submatrix, eliminating Python nested loops. Orientation is decided solely from node_1’s marginal gain against the current sets (sum to set_B vs sum to set_A); node_2 is forced to the opposite set to ensure the chosen edge contributes to the cut. This yields a best-by-edge-weight choice, not a best-improvement-by-cut-gain choice, and may accept non-improving steps relative to the current cut. Single-node remainder is handled by inserting that node into set A without evaluation, introducing a deterministic bias in the terminal step. Ties are resolved by NumPy’s argmax returning the first maximum in row-major order over the U-induced submatrix; the exact tie-breaking depends on the iteration order of the input set U. Supports asymmetric (directed) weight matrices since all gains are computed from the provided weights. Time complexity: O(|U|^2) for the submatrix argmax plus O(|A|+|B|) to compute the orientation; working memory: O(|U|^2) due to the submatrix.
    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): Edge weight matrix W of shape (n, n). Used as provided (supports directed/asymmetric weights).
            - "current_solution" (Solution): Current partition with attributes set_a and set_b.
            - "unselected_nodes" (set[int]): Nodes not yet assigned to either set.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this heuristic and returned unchanged.
        kwargs: Optional hyper-parameters accepted for interface compatibility but ignored in this variant:
            - gamma (float): Trade-off weight multiplying a balance bonus (unused). Default 0.1 if provided.
            - degree_power (float): Exponent applied to node weighted degrees in a bonus (unused). Default 1.0 if provided.
            - pair_scan_limit (int): Maximum number of unordered pairs evaluated (unused; full vectorized argmax is used). Default 0 if provided.
    Returns:
        InsertEdgeOperator: Operator that inserts the selected pair with orientation i→A, j→B or i→B, j→A, based on node_1’s marginal gain.
        InsertNodeOperator: If exactly one unselected node remains, inserts that node into set A.
        None: If there are no unselected nodes or no viable pair is found.
        dict: Unmodified algorithm_data, as this heuristic does not update it.
    """
    import numpy as np

    W = problem_state["weight_matrix"]
    sol = problem_state["current_solution"]
    U = problem_state["unselected_nodes"]

    # Handle trivial cases
    if not U:
        return None, algorithm_data
    if len(U) == 1:
        node = next(iter(U))
        return InsertNodeOperator(node=node, target_set='A'), algorithm_data

    # Build submatrix over unselected nodes and get global argmax excluding diagonal
    U_arr = np.fromiter(U, dtype=int)
    sub = W[np.ix_(U_arr, U_arr)].copy()
    # Exclude self-edges
    np.fill_diagonal(sub, -np.inf)
    flat_idx = int(np.argmax(sub))
    i_idx, j_idx = np.unravel_index(flat_idx, sub.shape)
    node_1 = int(U_arr[i_idx])
    node_2 = int(U_arr[j_idx])

    A = sol.set_a
    B = sol.set_b

    # Respect any existing assignments first
    if (node_1 in A) or (node_2 in B):
        op = InsertEdgeOperator(node_1=node_1, node_2=node_2)
        return op, algorithm_data
    if (node_1 in B) or (node_2 in A):
        op = InsertEdgeOperator(node_1=node_2, node_2=node_1)
        return op, algorithm_data

    # Compute node_1's marginal gains using NumPy vectorized sums
    gain_to_A = float(W[node_1, list(B)].sum()) if B else 0.0
    gain_to_B = float(W[node_1, list(A)].sum()) if A else 0.0

    if gain_to_A >= gain_to_B:
        op = InsertEdgeOperator(node_1=node_1, node_2=node_2)
    else:
        op = InsertEdgeOperator(node_1=node_2, node_2=node_1)
    return op, algorithm_data