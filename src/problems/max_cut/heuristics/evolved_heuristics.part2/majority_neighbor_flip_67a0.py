from src.problems.max_cut.components import *
import random

def majority_neighbor_flip_67a0(
    problem_state: dict,
    algorithm_data: dict,
    prefer_improving: bool = True,
    exploration_rate: float = 0.0,
    fallback_to_other_mode: bool = True,
    include_zero_delta: bool = False,
    **kwargs
) -> tuple[SwapOperator, dict]:
    """Randomized majority-neighbor single-node flip (1-opt local move for MaxCut).
    
    This heuristic evaluates the effect of flipping each already-assigned vertex to the opposite set by comparing:
    - same_side_sum: total weight from the vertex to vertices on its current side,
    - opposite_side_sum: total weight from the vertex to vertices on the opposite side.
    The cut gain delta of flipping equals (same_side_sum - opposite_side_sum). We build candidate pools:
    - Improving: delta > 0,
    - Worsening: delta < 0,
    - Neutral: delta == 0 (optional).
    The heuristic then samples one vertex from the chosen pool (improving by default, with optional exploration and fallbacks) and returns a SwapOperator to flip that vertex, preserving partition feasibility. This provides controlled stochastic local search with occasional diversification.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): Edge-weight matrix (assumed non-negative or arbitrary; symmetry not required by the code).
            - "current_solution" (Solution): Current partition, using sets current_solution.set_a and current_solution.set_b.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this heuristic.
        prefer_improving (bool): If True, sample from strictly improving candidates (delta > 0). If False, sample from strictly worsening candidates (delta < 0). Default is True.
        exploration_rate (float): Probability in [0.0, 1.0] to invert the selection mode for this call (e.g., pick worsening when prefer_improving=True). Enables occasional diversification. Default is 0.0.
        fallback_to_other_mode (bool): If the chosen candidate pool is empty, try the opposite pool before giving up. Default is True.
        include_zero_delta (bool): If True and both strict pools are empty, include zero-delta vertices as neutral candidates. Default is False.

    Returns:
        SwapOperator: An operator that flips exactly one assigned vertex to the opposite set. Returns None when no candidate exists (e.g., empty partition or all vertices yield no valid flip).
        dict: Lightweight metadata about the move: {"last_mode": str, "last_selected_node": int, "last_delta": float}. Returns {} if no operator is produced.
    """
    weight_matrix = problem_state.get("weight_matrix", None)
    current_solution = problem_state.get("current_solution", None)

    if weight_matrix is None or current_solution is None:
        return None, {}

    set_a = current_solution.set_a
    set_b = current_solution.set_b

    if (not set_a) and (not set_b):
        return None, {}

    improving_candidates = []
    worsening_candidates = []
    zero_candidates = []
    delta_by_node = {}

    assigned_nodes = set_a.union(set_b)
    list_a = list(set_a)
    list_b = list(set_b)

    for node in assigned_nodes:
        if node in set_a:
            same_side_sum = sum(weight_matrix[node, other] for other in list_a if other != node) if list_a else 0.0
            opposite_side_sum = sum(weight_matrix[node, other] for other in list_b) if list_b else 0.0
        else:
            same_side_sum = sum(weight_matrix[node, other] for other in list_b if other != node) if list_b else 0.0
            opposite_side_sum = sum(weight_matrix[node, other] for other in list_a) if list_a else 0.0

        delta = float(same_side_sum - opposite_side_sum)
        delta_by_node[node] = delta

        if delta > 0:
            improving_candidates.append(node)
        elif delta < 0:
            worsening_candidates.append(node)
        else:
            zero_candidates.append(node)

    use_improving = prefer_improving
    if exploration_rate > 0.0 and random.random() < exploration_rate:
        use_improving = not use_improving

    selected_list = improving_candidates if use_improving else worsening_candidates
    selected_mode = "improving" if use_improving else "worsening"

    if not selected_list and fallback_to_other_mode:
        selected_list = worsening_candidates if use_improving else improving_candidates
        selected_mode = "worsening" if use_improving else "improving"

    if not selected_list and include_zero_delta and zero_candidates:
        selected_list = zero_candidates
        selected_mode = "neutral"

    if not selected_list:
        return None, {}

    chosen_node = random.choice(list(selected_list))
    chosen_delta = delta_by_node.get(chosen_node, 0.0)

    op = SwapOperator([chosen_node])
    updated_info = {
        "last_mode": selected_mode,
        "last_selected_node": chosen_node,
        "last_delta": chosen_delta,
    }
    return op, updated_info