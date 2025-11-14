from src.problems.max_cut.components import *
import random
import numpy as np

def first_improvement_flip_7a32(problem_state: dict, algorithm_data: dict, epsilon: float=1e-12, scan_order: str="A_then_B", shuffle: bool=False, **kwargs) -> tuple[SwapOperator, dict]:
    """First-improvement single-node flip local search for undirected MaxCut.

    Scans currently assigned vertices and flips the first vertex whose move to the opposite set yields a strictly positive increase in the cut value. The flip gain (delta) is computed using pre-aggregated weights from each vertex to sets A and B:
      - If vertex ∈ A: delta = sum_w(vertex, A) − sum_w(vertex, B) for the A→B flip
      - If vertex ∈ B: delta = sum_w(vertex, B) − sum_w(vertex, A) for the B→A flip
    The heuristic follows a first-improvement policy (stops at the first improving vertex), supports configurable scan order, and optional shuffling to diversify descent trajectories. If no improving flip exists, returns (None, {}).

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): Symmetric adjacency/weight matrix of the undirected graph.
            - "current_solution" (Solution): Current partition with sets A and B.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this heuristic.
        epsilon (float): Strict positivity threshold for accepting a flip (delta > epsilon). Default is 1e-12.
        scan_order (str): Order to scan assigned nodes. Options: "A_then_B", "B_then_A", "interleaved". Default is "A_then_B".
        shuffle (bool): If True, randomly shuffles the final scan sequence before evaluation to reduce deterministic bias. Default is False.

    Returns:
        SwapOperator: Operator flipping a single improving vertex to the opposite set (first-improvement move).
        dict: Empty dictionary; no algorithm-specific data is updated.

        If no strictly improving flip exists or no nodes are assigned, returns (None, {}).
    """
    weight_matrix = problem_state["weight_matrix"]
    current_solution = problem_state["current_solution"]

    set_a = set(current_solution.set_a)
    set_b = set(current_solution.set_b)
    if not set_a and not set_b:
        return None, {}

    idx_a = list(set_a)
    idx_b = list(set_b)

    n = weight_matrix.shape[0]
    weight_to_a = np.zeros(n, dtype=weight_matrix.dtype) if len(idx_a) == 0 else weight_matrix[:, idx_a].sum(axis=1)
    weight_to_b = np.zeros(n, dtype=weight_matrix.dtype) if len(idx_b) == 0 else weight_matrix[:, idx_b].sum(axis=1)

    if scan_order == "A_then_B":
        scan_nodes = idx_a + idx_b
    elif scan_order == "B_then_A":
        scan_nodes = idx_b + idx_a
    elif scan_order == "interleaved":
        scan_nodes = []
        la, lb = len(idx_a), len(idx_b)
        for k in range(max(la, lb)):
            if k < la:
                scan_nodes.append(idx_a[k])
            if k < lb:
                scan_nodes.append(idx_b[k])
    else:
        scan_nodes = idx_a + idx_b

    if shuffle and scan_nodes:
        random.shuffle(scan_nodes)

    for node in scan_nodes:
        if node in set_a:
            delta = float(weight_to_a[node] - weight_to_b[node])
        elif node in set_b:
            delta = float(weight_to_b[node] - weight_to_a[node])
        else:
            continue

        if delta > epsilon:
            return SwapOperator([node]), {}

    return None, {}