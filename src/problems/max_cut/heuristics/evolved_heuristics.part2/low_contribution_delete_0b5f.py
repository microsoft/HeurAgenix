from src.problems.max_cut.components import *
import random
import math

def low_contribution_delete_0b5f(problem_state: dict, algorithm_data: dict, mode: str = "bottom_p", bottom_p: float = 0.2, worst_k: int = 1, seed: int = None, **kwargs) -> tuple[DeleteOperator, dict]:
    """Delete a low-contribution vertex to diversify the current partition.
    
    This perturbation targets vertices that contribute the least to the current cut (sum of weights to the opposite set),
    removing one to escape local minima and enable subsequent improvement steps. Edge weights are treated as undirected
    via symmetric averaging for robustness on asymmetric matrices. The deleted vertex is selected from a tail pool
    (either bottom fraction or worst-k) with uniform random tie-breaking, optionally seeded for reproducibility.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): 2D array of edge weights; treated as undirected via (w[i,j] + w[j,i]) / 2.
            - "current_solution" (Solution): Current partition containing set_a and set_b.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this heuristic for input.
            Returned data includes diagnostics for downstream algorithms:
            - "last_deleted_node" (int): Index of the deleted node.
            - "last_deleted_contribution" (float): Its cross-set contribution at selection time.
            - "candidate_pool_size" (int): Size of the tail pool used.
            - "selection_mode" (str): "bottom_p" or "worst_k".
        mode (str): Candidate pool construction strategy. 
            - "bottom_p": build pool from the worst p-fraction by contribution. Default is "bottom_p".
            - "worst_k": build pool from the k worst nodes by contribution.
        bottom_p (float): Fraction (0 < p ≤ 1) of assigned nodes to include when mode="bottom_p". Default is 0.2.
        worst_k (int): Number of worst nodes to include when mode="worst_k". Must be ≥ 1. Default is 1.
        seed (int | None): Random seed for reproducible selection from the candidate pool. Default is None.

    Returns:
        DeleteOperator: Operator that removes one selected node from its current set (A or B).
        dict: Diagnostics with the selected node and pool details. If no assigned nodes exist, returns None, {}.
    """
    weight_matrix = problem_state["weight_matrix"]
    current_solution = problem_state["current_solution"]
    set_a = current_solution.set_a
    set_b = current_solution.set_b

    assigned_nodes = list(set_a.union(set_b))
    if not assigned_nodes:
        return None, {}

    rng = random.Random(seed) if seed is not None else random

    def undirected_w(i: int, j: int) -> float:
        return 0.5 * (float(weight_matrix[i, j]) + float(weight_matrix[j, i]))

    contributions = []
    for node in assigned_nodes:
        others = set_b if node in set_a else set_a
        if not others:
            contrib = 0.0
        else:
            contrib = 0.0
            for j in others:
                contrib += undirected_w(node, j)
        contributions.append((node, contrib))

    contributions.sort(key=lambda x: x[1])

    if mode == "worst_k":
        k = max(1, int(worst_k))
        cand_size = min(k, len(contributions))
        selection_mode = "worst_k"
    else:
        p = max(0.0, min(1.0, float(bottom_p)))
        cand_size = max(1, int(math.floor(p * len(contributions)))) if p > 0.0 else 1
        selection_mode = "bottom_p"

    candidate_pool = contributions[:cand_size]
    if not candidate_pool:
        return None, {}

    node_to_delete, node_contrib = rng.choice(candidate_pool)
    op = DeleteOperator(node=node_to_delete)
    updated_data = {
        "last_deleted_node": node_to_delete,
        "last_deleted_contribution": node_contrib,
        "candidate_pool_size": len(candidate_pool),
        "selection_mode": selection_mode,
    }
    return op, updated_data