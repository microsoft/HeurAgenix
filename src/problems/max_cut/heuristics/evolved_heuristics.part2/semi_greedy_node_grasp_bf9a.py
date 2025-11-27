from src.problems.max_cut.components import *
import random

def semi_greedy_node_grasp_bf9a(problem_state: dict, algorithm_data: dict, rcl_size: int=5, random_seed: int=None, **kwargs) -> tuple[InsertNodeOperator, dict]:
    """GRASP-style semi-greedy constructive insertion for MaxCut.
    
    Builds a Restricted Candidate List (RCL) of unassigned vertices ranked by their best insertion gain,
    where a vertex’s gain is max(ΔA, ΔB) with:
      - ΔA: sum of weights from the vertex to vertices currently in set B (gain if inserted into A),
      - ΔB: sum of weights from the vertex to vertices currently in set A (gain if inserted into B).
    One vertex is uniformly sampled from the top-rcl_size candidates and inserted into the side that
    realizes its larger gain (ties resolved to A). This balances exploitation (top gains) and exploration
    (random choice within the RCL). Time per call is O(|V_unselected| · (|A| + |B|)) for gain computation.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): Edge weight matrix used to compute per-vertex insertion gains.
            - "current_solution" (Solution): Current bipartition with sets set_a and set_b; read-only here.
            - "unselected_nodes" (set[int]): Vertices not assigned to either set; candidate pool for insertion.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this heuristic for input.
            Returned data includes selection trace for downstream analytics.
        rcl_size (int): Size of the Restricted Candidate List (top candidates by gain). Must be >= 1. Default is 5.
        random_seed (int | None): Seed to make the RCL sampling reproducible. If None, uses global randomness. Default is None.

    Returns:
        InsertNodeOperator: Inserts the sampled vertex into the side (A or B) that yields the larger gain.
        dict: Updated algorithm data containing:
              - "rcl_size" (int): Effective RCL size used.
              - "random_seed" (int | None): Seed used for sampling.
              - "last_selected_node" (int): The vertex chosen from the RCL.
              - "last_selected_gain" (float): The gain associated with the chosen side.
              - "last_target_set" (str): 'A' or 'B', the side chosen.
        If there are no unselected nodes, returns (None, {}) to indicate no constructive move is possible.
    """
    weight_matrix = problem_state["weight_matrix"]
    current_solution = problem_state["current_solution"]
    unselected_nodes = problem_state["unselected_nodes"]

    # If no candidates remain, fail gracefully.
    if not unselected_nodes:
        return None, {}

    # Validate and clamp RCL size.
    if not isinstance(rcl_size, int) or rcl_size < 1:
        rcl_size = 1

    set_a = current_solution.set_a
    set_b = current_solution.set_b

    # Compute best insertion gain and target side for each unselected node.
    candidates = []
    for node in unselected_nodes:
        delta_a = sum(weight_matrix[node, other] for other in set_b) if set_b else 0
        delta_b = sum(weight_matrix[node, other] for other in set_a) if set_a else 0

        if delta_a >= delta_b:
            candidates.append((node, delta_a, 'A'))
        else:
            candidates.append((node, delta_b, 'B'))

    if not candidates:
        return None, {}

    # Form RCL from top-gain candidates.
    candidates.sort(key=lambda x: x[1], reverse=True)
    rcl_len = min(rcl_size, len(candidates))
    rcl = candidates[:rcl_len]

    # Sample uniformly from the RCL; use a local RNG if a seed is provided.
    if random_seed is None:
        sampled_idx = random.randrange(rcl_len)
    else:
        rng = random.Random(random_seed)
        sampled_idx = rng.randrange(rcl_len)

    selected_node, selected_gain, selected_side = rcl[sampled_idx]

    operator = InsertNodeOperator(node=selected_node, target_set=selected_side)

    updated_data = {
        "rcl_size": rcl_size,
        "random_seed": random_seed,
        "last_selected_node": selected_node,
        "last_selected_gain": float(selected_gain),
        "last_target_set": selected_side,
    }
    return operator, updated_data