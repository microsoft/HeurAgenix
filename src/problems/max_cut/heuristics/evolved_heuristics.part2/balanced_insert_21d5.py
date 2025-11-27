from src.problems.max_cut.components import *
import random

def balanced_insert_21d5(problem_state: dict, algorithm_data: dict, tie_bias: str='A', pick_strategy: str='min', **kwargs) -> tuple[InsertNodeOperator, dict]:
    """Balance-driven single-node insertion with deterministic or random vertex selection. Chooses one unassigned vertex (by default the smallest-index for determinism, optionally random) and places it into the smaller partition to reduce size imbalance; when both partitions have equal cardinality, the tie is resolved by the configurable bias (A or B). This heuristic is constructive and weight-agnostic: it does not evaluate cut gain, focusing purely on set-size balance. Suitable for building initial feasible partitions or rebalancing during diversification stages. Processes exactly one vertex per call.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current partition with sets A and B used to compute set sizes.
            - "unselected_nodes" (set[int]): Nodes not yet assigned to either set, eligible for insertion.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this heuristic.
        tie_bias (str): The target set used when |A| == |B|. Allowed values: 'A' or 'B'. Default is 'A'.
        pick_strategy (str): Strategy to select the node from unselected_nodes.
            - 'min': pick the smallest node id for determinism.
            - 'random': uniformly sample one node (requires 'random' import).
            Default is 'min'.

    Returns:
        InsertNodeOperator: Operator that inserts the chosen node into the smaller set (or the tie-bias set if equal).
        dict: Empty dictionary as no algorithm data is updated.
    """
    current_solution = problem_state['current_solution']
    unselected_nodes = problem_state['unselected_nodes']

    # If there are no unselected nodes left, return None.
    if not unselected_nodes:
        return None, {}

    # Select node per strategy
    if pick_strategy == 'random':
        node_to_insert = random.choice(list(unselected_nodes))
    else:
        # Default to deterministic smallest id to avoid set iteration order dependence
        node_to_insert = min(unselected_nodes)

    # Compute cardinalities
    set_a_count = len(current_solution.set_a)
    set_b_count = len(current_solution.set_b)

    # Resolve tie-bias safely
    bias = tie_bias if tie_bias in ('A', 'B') else 'A'

    # Decide target set based on balance
    if set_a_count < set_b_count:
        target_set = 'A'
    elif set_a_count > set_b_count:
        target_set = 'B'
    else:
        target_set = bias

    operator = InsertNodeOperator(node=node_to_insert, target_set=target_set)
    return operator, {}