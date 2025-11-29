from src.problems.max_cut.components import *
import heapq
import random
import numpy as np

def heaviest_edge_seed_eb0d(
    problem_state: dict,
    algorithm_data: dict,
    tie_break_random: bool=False,
    orientation_mode: str='node1_marginal',
    single_node_target: str='A',
    **kwargs
):
    """
    Greedy constructive seeding by globally heaviest unselected directed edge (fast, cached).
    Among all pairs (i, j) with i != j in the unselected set, select the edge with maximum weight_matrix[i][j].
    This fast variant avoids O(k^2) scans by pre-sorting each row once and maintaining a max-heap of each unselected row’s current
    best unselected neighbor (lazy invalidation). Selection becomes amortized O(log n) per call after O(n^2 log n) preprocessing.

    Orientation (which node goes to A vs B) is decided by:
      - Default: node1_marginal — compare node_1’s outgoing sum to current set B (as A placement) vs to set A (as B placement),
        computed via vectorized numpy summations over the current sets.
      - Optional: pair_gain — compare gain_AB = sum_to_B[node_1] + sum_to_A[node_2] versus gain_BA = sum_to_A[node_1] + sum_to_B[node_2].
        The pair’s mutual directed contributions cancel and need not be computed explicitly.
    If exactly one unselected node remains, insert it into a specified target side (single_node_target). Ties among edges are optionally
    broken uniformly at random among rows’ current tops with equal maximum weight (no full pair scan). Supports asymmetric (directed)
    weight matrices; all computations use the given directed weights.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): 2D adjacency/weight matrix; weight_matrix[i][j] is the directed weight from i to j.
            - "current_solution" (Solution): Current partition with sets 'set_a' and 'set_b'.
            - "unselected_nodes" (set[int]): Nodes not yet assigned to either set.
        algorithm_data (dict): Used to cache preprocessing state across calls for speed:
            - Stores per-row descending neighbor orders (excluding self), row pointers to the current best unselected neighbor,
              and a max-heap of current row tops. Automatically rebuilt if the weight matrix object or problem size changes.

        tie_break_random (bool): If True, break ties among equally heaviest edges uniformly at random (among current row tops).
            Default is False.
        orientation_mode (str): Orientation rule for the selected edge.
            - 'node1_marginal': Place node_1 to the side that maximizes its outgoing contribution (sum to opposite set).
            - 'pair_gain': Compare two orientations by summing outgoing contributions of both nodes to their opposite sets
              using directed outgoing sums (pair terms cancel).
            Default is 'node1_marginal'.
        single_node_target (str): Target side for inserting the last remaining single node. Allowed values: 'A' or 'B'.
            If the node is already in a set, its existing assignment is respected to avoid assertion errors.
            Default is 'A'.
    Returns:
        InsertEdgeOperator: Operator to add the chosen heaviest edge endpoints, placing node_1 into set A and node_2 into set B
        according to the selected orientation rule (and respecting existing assignments if any).
        dict: Empty dictionary as no algorithm data is updated.
    """
    W = problem_state["weight_matrix"]
    sol = problem_state["current_solution"]
    set_a = sol.set_a
    set_b = sol.set_b
    unselected = problem_state["unselected_nodes"]

    n = W.shape[0]
    if not unselected:
        return None, algorithm_data

    # Single node remains: insert to configured side (respecting assignment if any)
    if len(unselected) == 1:
        single = next(iter(unselected))
        if single in set_a:
            target_set = 'A'
        elif single in set_b:
            target_set = 'B'
        else:
            target_set = 'A' if single_node_target == 'A' else 'B'
        return InsertNodeOperator(node=single, target_set=target_set), algorithm_data

    # Initialize or reuse cached state
    state_key = "heaviest_edge_seed_eb0c_fast"
    state = algorithm_data.get(state_key)

    if (state is None) or (state.get("n") != n) or (state.get("W_id") != id(W)):
        # Build per-row descending neighbor order excluding self
        row_order = []
        for i in range(n):
            order = np.argsort(-W[i], kind='quicksort')
            order = order[order != i]
            row_order.append(order)
        ptr = np.zeros(n, dtype=np.int32)
        heap = []

        # Boolean mask for quick membership checks
        is_unselected = np.zeros(n, dtype=bool)
        is_unselected[list(unselected)] = True

        # Initialize heap with top unselected neighbor per unselected row
        for i in range(n):
            if not is_unselected[i]:
                continue
            order = row_order[i]
            p = 0
            m = order.size
            while p < m and not is_unselected[order[p]]:
                p += 1
            ptr[i] = p
            if p < m:
                j = int(order[p])
                w = W[i, j]
                heapq.heappush(heap, (-w, i, j))

        state = {
            "row_order": row_order,
            "ptr": ptr,
            "heap": heap,
            "n": n,
            "W_id": id(W),
        }
    else:
        row_order = state["row_order"]
        ptr = state["ptr"]
        heap = state["heap"]

    # Rebuild membership mask for current call (nodes may have been assigned since last call)
    is_unselected = np.zeros(n, dtype=bool)
    is_unselected[list(unselected)] = True

    def ensure_row_top(i: int):
        if not is_unselected[i]:
            return
        order = row_order[i]
        p = ptr[i]
        m = order.size
        while p < m and not is_unselected[order[p]]:
            p += 1
        if p != ptr[i]:
            ptr[i] = p
            if p < m:
                j = int(order[p])
                w = W[i, j]
                heapq.heappush(heap, (-w, i, j))

    def pop_valid_top():
        while heap:
            negw, i, j = heapq.heappop(heap)
            if not is_unselected[i]:
                continue
            # Align row pointer to current top j
            order = row_order[i]
            p = ptr[i]
            m = order.size
            while p < m and not is_unselected[order[p]]:
                p += 1
            if p != ptr[i]:
                ptr[i] = p
            if p < m:
                j2 = int(order[p])
                w2 = W[i, j2]
                if j2 != j or -negw != w2:
                    # Stale entry; push fresh and continue
                    heapq.heappush(heap, (-w2, i, j2))
                    continue
                return w2, i, j2
        return None

    # Get current best
    top = pop_valid_top()
    if top is None:
        # No available directed edge among unselected (should be rare)
        new_alg = dict(algorithm_data)
        new_alg[state_key] = state
        return None, new_alg
    w_star, i_star, j_star = top

    # Optional random tie-breaking among rows’ current tops with same weight
    if tie_break_random:
        candidates = [(i_star, j_star)]
        # Make sure every unselected row’s pointer is aligned and heap contains its top
        for i in np.nonzero(is_unselected)[0]:
            if i == i_star:
                continue
            ensure_row_top(int(i))
            p = ptr[int(i)]
            order = row_order[int(i)]
            if p < order.size:
                j = int(order[p])
                if W[int(i), j] == w_star:
                    candidates.append((int(i), j))
        i_sel, j_sel = random.choice(candidates)
    else:
        i_sel, j_sel = i_star, j_star

    # Orientation decision
    # Respect existing assignments first to avoid assertion violations in InsertEdgeOperator
    if (i_sel in set_a) or (j_sel in set_b):
        op = InsertEdgeOperator(node_1=i_sel, node_2=j_sel)
    elif (i_sel in set_b) or (j_sel in set_a):
        op = InsertEdgeOperator(node_1=j_sel, node_2=i_sel)
    else:
        if orientation_mode == 'pair_gain':
            # Compare gain_AB vs gain_BA (pair contributions cancel)
            if set_b:
                i_to_B = float(W[i_sel, list(set_b)].sum())
                j_to_B = float(W[j_sel, list(set_b)].sum())
            else:
                i_to_B = 0.0
                j_to_B = 0.0
            if set_a:
                i_to_A = float(W[i_sel, list(set_a)].sum())
                j_to_A = float(W[j_sel, list(set_a)].sum())
            else:
                i_to_A = 0.0
                j_to_A = 0.0
            gain_AB = i_to_B + j_to_A
            gain_BA = i_to_A + j_to_B
            if gain_AB >= gain_BA:
                op = InsertEdgeOperator(node_1=i_sel, node_2=j_sel)
            else:
                op = InsertEdgeOperator(node_1=j_sel, node_2=i_sel)
        else:
            # node1_marginal
            i_to_B = float(W[i_sel, list(set_b)].sum()) if set_b else 0.0
            i_to_A = float(W[i_sel, list(set_a)].sum()) if set_a else 0.0
            if i_to_B >= i_to_A:
                op = InsertEdgeOperator(node_1=i_sel, node_2=j_sel)
            else:
                op = InsertEdgeOperator(node_1=j_sel, node_2=i_sel)

    new_alg = dict(algorithm_data)
    new_alg[state_key] = state
    return op, new_alg