from src.problems.max_cut.components import *
import random
import math
import numpy as np

def heavy_edge_matching_seed_edd4(
    problem_state: dict,
    algorithm_data: dict,
    pair_sample_ratio: float = 1.0,
    max_pair_evaluations: int = None,
    seed: int = None,
    **kwargs
) -> tuple[InsertEdgeOperator, dict]:
    """Greedy heavy-edge matching seed (HMS) for MaxCut with optional partial scanning.
    
    This constructive heuristic selects, among currently unassigned vertices, the heaviest edge (by weight) and inserts
    both endpoints into opposite sets to maximize the immediate cut gain against the existing partition. Orientation
    (i→A, j→B vs. i→B, j→A) is chosen by evaluating each endpoint’s aggregated connectivity to the current opposite sets.
    When exactly one vertex remains unassigned, the heuristic inserts it alone into the more beneficial side. For large
    instances, partial scanning is supported via a pair sampling ratio and/or a hard cap on pair evaluations. An optional
    RNG seed diversifies the scan order under partial evaluation. The approach is well-suited as a seeding step before
    local search or as a constructive move in iterative improvement/metaheuristics.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): Symmetric (or given as-is) adjacency matrix of edge weights.
            - "current_solution" (Solution): Current partition; uses current_solution.set_a and current_solution.set_b.
            - "unselected_nodes" (set[int]): Vertices not yet placed in either set; only these are considered.
        algorithm_data (dict): The algorithm dictionary for this heuristic only. Not used in this heuristic.
        pair_sample_ratio (float): Fraction of all unordered unselected pairs to evaluate. Default is 1.0.
            - Range: (0, 1] for partial scan; 1.0 for full scan. Values ≤ 0 are clamped to evaluate at least one pair.
        max_pair_evaluations (int | None): Optional hard cap on the number of pairs evaluated. Default is None (no cap).
            - If provided and > 0, it overrides the budget computed from pair_sample_ratio when smaller.
        seed (int | None): RNG seed used to randomize the pair scan order (via shuffling unselected nodes). Default is None.
            - Only impacts order under partial scanning/budgeted evaluations; full scans remain exhaustive.

    Returns:
        InsertEdgeOperator: Inserts two endpoints of the selected heaviest edge into opposite sets with orientation maximizing immediate gain.
            - Corner case: if exactly one node remains unselected, returns an InsertNodeOperator placing it into the better side.
        dict: Updated summary data for downstream use:
            {
                "last_selected_pair": (i, j) or None for single-node case,
                "evaluated_pairs": int,
                "pair_sample_ratio": float,
                "max_pair_evaluations": int | None,
                "seed_used": int | None
            }
    """
    weight_matrix = problem_state["weight_matrix"]
    current_solution = problem_state["current_solution"]
    unselected_nodes = problem_state["unselected_nodes"]

    # If there are no unselected nodes left, return None (no feasible insertion).
    if not unselected_nodes:
        return None, {}

    # Single-node remainder: insert into the side that maximizes immediate cut gain.
    if len(unselected_nodes) == 1:
        lone = next(iter(unselected_nodes))
        gain_to_a = sum(weight_matrix[lone, b] for b in current_solution.set_b) if current_solution.set_b else 0.0
        gain_to_b = sum(weight_matrix[lone, a] for a in current_solution.set_a) if current_solution.set_a else 0.0
        target_set = 'A' if gain_to_a >= gain_to_b else 'B'
        return InsertNodeOperator(node=lone, target_set=target_set), {
            "last_selected_pair": None,
            "evaluated_pairs": 0,
            "pair_sample_ratio": pair_sample_ratio,
            "max_pair_evaluations": max_pair_evaluations,
            "seed_used": seed
        }

    # Precompute aggregated connectivity to current sets for O(1) gain checks.
    set_a_list = list(current_solution.set_a)
    set_b_list = list(current_solution.set_b)
    weight_to_a = weight_matrix[:, set_a_list].sum(axis=1) if set_a_list else np.zeros(weight_matrix.shape[0], dtype=float)
    weight_to_b = weight_matrix[:, set_b_list].sum(axis=1) if set_b_list else np.zeros(weight_matrix.shape[0], dtype=float)

    # Prepare unordered pair iteration over unselected nodes.
    unselected_list = list(unselected_nodes)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(unselected_list)  # randomize scan order to diversify near-max selection under partial budget
    else:
        unselected_list.sort()  # deterministic order when not randomized

    n_u = len(unselected_list)
    total_pairs = n_u * (n_u - 1) // 2

    # Determine evaluation budget (partial scan support).
    if pair_sample_ratio <= 0.0:
        eval_limit = 1
    else:
        eval_limit = int(math.ceil(pair_sample_ratio * total_pairs))
        eval_limit = max(1, min(eval_limit, total_pairs))
    if isinstance(max_pair_evaluations, int) and max_pair_evaluations > 0:
        eval_limit = min(eval_limit, max_pair_evaluations)

    # Scan up to eval_limit pairs to select the single heaviest edge.
    best_w = -float("inf")
    best_i = None
    best_j = None
    evaluated = 0

    for idx_i in range(n_u):
        if evaluated >= eval_limit:
            break
        i = unselected_list[idx_i]
        for idx_j in range(idx_i + 1, n_u):
            j = unselected_list[idx_j]
            w_ij = float(weight_matrix[i, j])
            if w_ij > best_w:
                best_w = w_ij
                best_i = i
                best_j = j
            evaluated += 1
            if evaluated >= eval_limit:
                break

    # Safety: if no pair evaluated (should not occur due to clamping), return None.
    if best_i is None or best_j is None:
        return None, {}

    # Choose orientation to maximize immediate gain relative to current partition.
    gain_AB = float(weight_to_b[best_i]) + float(weight_to_a[best_j]) + best_w  # i→A, j→B
    gain_BA = float(weight_to_a[best_i]) + float(weight_to_b[best_j]) + best_w  # i→B, j→A

    # Deterministic tie-break toward (i→A, j→B).
    if gain_AB >= gain_BA:
        operator = InsertEdgeOperator(node_1=best_i, node_2=best_j)
    else:
        operator = InsertEdgeOperator(node_1=best_j, node_2=best_i)

    return operator, {
        "last_selected_pair": (best_i, best_j),
        "evaluated_pairs": evaluated,
        "pair_sample_ratio": pair_sample_ratio,
        "max_pair_evaluations": max_pair_evaluations,
        "seed_used": seed
    }