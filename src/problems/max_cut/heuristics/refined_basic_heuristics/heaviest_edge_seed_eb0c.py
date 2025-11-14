from src.problems.max_cut.components import *
import random

def heaviest_edge_seed_eb0c(
    problem_state: dict,
    algorithm_data: dict,
    tie_break_random: bool=False,
    orientation_mode: str='node1_marginal',
    single_node_target: str='A',
    **kwargs
) -> tuple[InsertEdgeOperator, dict]:
    """Greedy constructive seeding by globally heaviest unselected directed edge.
    Among all pairs (i, j) with i != j in the unselected set, select the edge with maximum weight_matrix[i][j].
    Orientation (which node goes to A vs B) is decided by:
      - Default: node1_marginal — compare node_1’s outgoing sum to current set B (as A placement) vs to set A (as B placement).
      - Optional: pair_gain — evaluate combined immediate cut contribution of both orientations using directed outgoing edges.
    If exactly one unselected node remains, insert it into a specified target side (single_node_target). Ties among edges are optionally
    broken uniformly at random. Supports asymmetric (directed) weight matrices; all computations use the given directed weights.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): 2D adjacency/weight matrix; weight_matrix[i][j] is the directed weight from i to j.
            - "current_solution" (Solution): Current partition with sets 'set_a' and 'set_b'.
            - "unselected_nodes" (set[int]): Nodes not yet assigned to either set.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this heuristic.

        tie_break_random (bool): If True, break ties among equally heaviest edges uniformly at random. Default is False.
        orientation_mode (str): Orientation rule for the selected edge.
            - 'node1_marginal': Place node_1 to the side that maximizes its outgoing contribution (sum to opposite set).
            - 'pair_gain': Compare two orientations by summing outgoing contributions of both nodes to their opposite sets,
              including both directed edge contributions between the pair.
            Default is 'node1_marginal'.
        single_node_target (str): Target side for inserting the last remaining single node. Allowed values: 'A' or 'B'.
            If the node is already in a set, its existing assignment is respected to avoid assertion errors.
            Default is 'A'.

    Returns:
        InsertEdgeOperator: Operator to add the chosen heaviest edge endpoints, placing node_1 into set A and node_2 into set B
        according to the selected orientation rule (and respecting existing assignments if any).
        dict: Empty dictionary as no algorithm data is updated.
    """
    weight_matrix = problem_state["weight_matrix"]
    current_solution = problem_state["current_solution"]
    unselected = problem_state["unselected_nodes"]

    # Boundary: No available nodes.
    if not unselected:
        return None, {}

    # Boundary: Single node remains — insert without edge evaluation.
    if len(unselected) == 1:
        single = next(iter(unselected))
        # Respect existing assignment if any; otherwise use the configured target.
        if single in current_solution.set_a:
            target_set = 'A'
        elif single in current_solution.set_b:
            target_set = 'B'
        else:
            target_set = 'A' if single_node_target == 'A' else 'B'
        return InsertNodeOperator(node=single, target_set=target_set), {}

    # Find the heaviest directed edge among unselected nodes.
    max_w = float('-inf')
    candidates = []
    for i in unselected:
        for j in unselected:
            if i == j:
                continue
            w = weight_matrix[i][j]
            if w > max_w:
                max_w = w
                candidates = [(i, j)]
            elif w == max_w:
                candidates.append((i, j))

    # If no candidate edge found, no move can be made.
    if not candidates:
        return None, {}

    # Optional random tie-breaking.
    node_1, node_2 = (random.choice(candidates) if tie_break_random else candidates[0])

    set_a = current_solution.set_a
    set_b = current_solution.set_b

    # Respect existing assignments to avoid operator assertion violations.
    if (node_1 in set_a) or (node_2 in set_b):
        return InsertEdgeOperator(node_1=node_1, node_2=node_2), {}
    if (node_1 in set_b) or (node_2 in set_a):
        return InsertEdgeOperator(node_1=node_2, node_2=node_1), {}

    # Decide orientation by the selected mode.
    if orientation_mode == 'pair_gain':
        # Combined immediate outgoing contribution for both orientations (directed).
        gain_AB = (
            sum(weight_matrix[node_1][x] for x in set_b) +  # node_1 -> B when node_1 in A
            sum(weight_matrix[node_2][x] for x in set_a) +  # node_2 -> A when node_2 in B
            weight_matrix[node_1][node_2] +                 # pair edge i->j crosses A->B
            weight_matrix[node_2][node_1]                   # pair edge j->i crosses B->A
        )
        gain_BA = (
            sum(weight_matrix[node_1][x] for x in set_a) +  # node_1 -> A when node_1 in B
            sum(weight_matrix[node_2][x] for x in set_b) +  # node_2 -> B when node_2 in A
            weight_matrix[node_1][node_2] +                 # pair edge still crosses sets
            weight_matrix[node_2][node_1]
        )
        if gain_AB >= gain_BA:
            return InsertEdgeOperator(node_1=node_1, node_2=node_2), {}
        else:
            return InsertEdgeOperator(node_1=node_2, node_2=node_1), {}
    else:
        # Default: node1_marginal — decide by node_1's outgoing marginal to opposite sets.
        marg_to_B = sum(weight_matrix[node_1][x] for x in set_b)  # placing node_1 in A
        marg_to_A = sum(weight_matrix[node_1][x] for x in set_a)  # placing node_1 in B
        if marg_to_B >= marg_to_A:
            return InsertEdgeOperator(node_1=node_1, node_2=node_2), {}
        else:
            return InsertEdgeOperator(node_1=node_2, node_2=node_1), {}