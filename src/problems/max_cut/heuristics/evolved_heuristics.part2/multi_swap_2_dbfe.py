from src.problems.max_cut.components import *
import random
import numpy as np

def multi_swap_2_dbfe(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[SwapOperator, dict]:
    """
    Best-improvement 2-swap (pairwise exchange) local search. Evaluates every (i ∈ A, j ∈ B) and applies the single best pair with strictly positive gain. Gain is computed from precomputed side-sum vectors:
Δ(i,j) = (ΣA w(i,·) − ΣB w(i,·)) + (ΣB w(j,·) − ΣA w(j,·)) + 2·w(i,j).
The +2·w(i,j) term corrects the double subtraction of the edge (i,j) when summing single-node flip gains, ensuring the edge’s contribution remains unchanged after a simultaneous swap. Side sums are vectorized via W[:,A]·1 and W[:,B]·1, yielding O(1) evaluation per pair after O(n|A| + n|B|) preprocessing. Selection is global best (not first-improvement) and strictly improving (no worsening acceptance). Assumes an undirected/symmetric weight matrix; otherwise the interaction term should use w(i,j)+w(j,i). Time complexity: O(n|A| + n|B| + |A|·|B|); O(n) extra memory. Suitable for escaping 1-flip local optima in a KL-style refinement step.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): A 2D array representing the weight between nodes.
            - "current_solution" (Solution): The current solution of the Max Cut problem.
        algorithm_data (dict): Not used in this heuristic.

    Returns:
        SwapOperator: The operator that swaps a pair of nodes between sets to improve the cut value.
        dict: Empty dictionary as no algorithm data is updated.
    """

    current_solution = problem_state['current_solution']
    weight_matrix = problem_state['weight_matrix']
    best_increase = 0
    best_pairs = []

    set_a = current_solution.set_a
    set_b = current_solution.set_b

    # Precompute the sum of weights to and from each node
    
    list_a = list(set_a)
    list_b = list(set_b)
    
    if not list_a or not list_b:
        return None, {}
        
    weight_to_a = weight_matrix[:, list_a].sum(axis=1)
    weight_to_b = weight_matrix[:, list_b].sum(axis=1)

    # Vectorized calculation
    # gain_a[i] = weight_to_a[i] - weight_to_b[i] (for i in A)
    # gain_b[j] = weight_to_b[j] - weight_to_a[j] (for j in B)
    
    gain_a_vals = weight_to_a[list_a] - weight_to_b[list_a]
    gain_b_vals = weight_to_b[list_b] - weight_to_a[list_b]
    
    # Interaction term: 2 * W[i, j]
    # We need submatrix W[list_a, list_b]
    # W_sub[k, l] corresponds to W[list_a[k], list_b[l]]
    W_sub = weight_matrix[np.ix_(list_a, list_b)]
    
    # Total delta matrix
    # shape: (len(A), len(B))
    # broadcasting: (len(A), 1) + (1, len(B)) + (len(A), len(B))
    delta_matrix = gain_a_vals[:, None] + gain_b_vals[None, :] + 2 * W_sub
    
    # Find max
    max_delta = np.max(delta_matrix)
    
    if max_delta > 0:
        # Find all pairs with max_delta
        best_indices = np.argwhere(delta_matrix == max_delta)
        # Randomly choose one
        chosen_idx = random.choice(best_indices)
        
        i = list_a[chosen_idx[0]]
        j = list_b[chosen_idx[1]]
        return SwapOperator([i, j]), {}
    else:
        return None, {}