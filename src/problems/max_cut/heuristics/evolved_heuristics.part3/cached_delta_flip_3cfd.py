from src.problems.max_cut.components import *
import numpy as np

def cached_delta_flip_3cfd(
    problem_state: dict,
    algorithm_data: dict,
    min_improvement: float = 1e-12,
    rebuild_on_signature_mismatch: bool = True,
    use_numpy_vectorization: bool = True,
    **kwargs
) -> tuple[SwapOperator, dict]:
    """Single-node flip with cached delta updates for MaxCut (incremental gain evaluation).
    
    Unique idea:
    - Maintain two cached arrays: weight_to_a[v] and weight_to_b[v], representing the sum of weights from node v to nodes in
      set A and set B, respectively. The cut-gain of flipping a node i is computed in O(1):
        * if i ∈ A: delta = weight_to_a[i] - weight_to_b[i]
        * if i ∈ B: delta = weight_to_b[i] - weight_to_a[i]
      A strictly positive delta indicates an improving move.
    - Caches are rebuilt on the first call or when the partition changes externally (detected via a signature).
    - After applying the best flip, caches are lazily updated in O(n) using only the column of weights to the flipped node
      (no full recomputation), enabling efficient repeated calls.
    
    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): n×n symmetric matrix of edge weights (undirected graph).
            - "current_solution" (Solution): Current partition with two disjoint sets: set_a and set_b.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary / optionally used:
            - "weight_to_a" (list[float]): Cached sum of weights from each node to set A (length n). Recomputed if missing or invalid.
            - "weight_to_b" (list[float]): Cached sum of weights from each node to set B (length n). Recomputed if missing or invalid.
            - "partition_signature" (tuple[tuple[int, ...], tuple[int, ...]]): Signature of the current partition (sorted sets) to detect external changes.
            - "last_move_node" (int): Not required. If present, overwritten with the last flipped node.
            - "last_delta" (float): Not required. If present, overwritten with the last flip gain.
        Hyperparameters:
            - min_improvement (float): Strict positivity threshold for accepting a flip (delta > min_improvement).
              Default is 1e-12. Set to 0.0 to allow non-decreasing flips.
            - rebuild_on_signature_mismatch (bool): Rebuild caches if the stored signature differs from the current partition.
              Default is True. Ensures correctness when other heuristics modify the solution.
            - use_numpy_vectorization (bool): Use NumPy vectorization to build caches; otherwise use pure-Python loops.
              Default is True.
    
    Returns:
        SwapOperator: An operator that flips exactly one node (A↔B) if a strictly improving move is found.
        dict: Updated algorithm data containing lazily updated caches and the new partition signature. If no improving
        flip exists, returns (None, updated_data) so caches remain available for subsequent calls.
    """
    # Read problem state
    weight_matrix = problem_state["weight_matrix"]
    current_solution = problem_state["current_solution"]
    set_a = current_solution.set_a
    set_b = current_solution.set_b

    n = int(weight_matrix.shape[0])

    # Build current partition signature
    partition_signature = (tuple(sorted(set_a)), tuple(sorted(set_b)))

    # Retrieve caches
    cached_a = algorithm_data.get("weight_to_a")
    cached_b = algorithm_data.get("weight_to_b")
    cached_sig = algorithm_data.get("partition_signature")

    # Decide whether to rebuild caches
    need_rebuild = (
        (cached_a is None) or
        (cached_b is None) or
        (len(cached_a) != n) or
        (len(cached_b) != n) or
        (rebuild_on_signature_mismatch and (cached_sig != partition_signature))
    )

    # (Re)build caches
    if need_rebuild:
        if use_numpy_vectorization:
            if set_a:
                weight_to_a = weight_matrix[:, list(set_a)].sum(axis=1).astype(float).tolist()
            else:
                weight_to_a = [0.0] * n
            if set_b:
                weight_to_b = weight_matrix[:, list(set_b)].sum(axis=1).astype(float).tolist()
            else:
                weight_to_b = [0.0] * n
        else:
            weight_to_a = [0.0] * n
            weight_to_b = [0.0] * n
            if set_a:
                for v in range(n):
                    s = 0.0
                    for u in set_a:
                        s += float(weight_matrix[v, u])
                    weight_to_a[v] = s
            if set_b:
                for v in range(n):
                    s = 0.0
                    for u in set_b:
                        s += float(weight_matrix[v, u])
                    weight_to_b[v] = s
    else:
        weight_to_a = list(cached_a)
        weight_to_b = list(cached_b)

    # Scan for the best strictly improving single-node flip
    best_node = None
    best_delta = 0.0

    # Evaluate nodes in set A: flip A -> B
    for i in set_a:
        delta = weight_to_a[i] - weight_to_b[i]
        if delta > best_delta:
            best_delta = delta
            best_node = i

    # Evaluate nodes in set B: flip B -> A
    for j in set_b:
        delta = weight_to_b[j] - weight_to_a[j]
        if delta > best_delta:
            best_delta = delta
            best_node = j

    # If no improving move exists, return updated caches without an operator
    if best_node is None or best_delta <= float(min_improvement):
        updated_data = {
            "weight_to_a": weight_to_a,
            "weight_to_b": weight_to_b,
            "partition_signature": partition_signature,
            "last_move_node": None,
            "last_delta": float(best_delta),
            "min_improvement": float(min_improvement),
        }
        return None, updated_data

    # Prepare the operator: flip the best node
    op = SwapOperator([best_node])

    # Lazily update caches in O(n) using the column to the flipped node
    updated_weight_to_a = list(weight_to_a)
    updated_weight_to_b = list(weight_to_b)
    column_to_node = weight_matrix[:, best_node]

    if best_node in set_a:
        # Move A -> B
        for v in range(n):
            w = float(column_to_node[v])
            updated_weight_to_a[v] -= w
            updated_weight_to_b[v] += w
        new_set_a = set_a - {best_node}
        new_set_b = set_b | {best_node}
    else:
        # Move B -> A
        for v in range(n):
            w = float(column_to_node[v])
            updated_weight_to_a[v] += w
            updated_weight_to_b[v] -= w
        new_set_a = set_a | {best_node}
        new_set_b = set_b - {best_node}

    new_signature = (tuple(sorted(new_set_a)), tuple(sorted(new_set_b)))

    # Return operator and updated caches
    updated_data = {
        "weight_to_a": updated_weight_to_a,
        "weight_to_b": updated_weight_to_b,
        "partition_signature": new_signature,
        "last_move_node": best_node,
        "last_delta": float(best_delta),
        "min_improvement": float(min_improvement),
    }
    return op, updated_data