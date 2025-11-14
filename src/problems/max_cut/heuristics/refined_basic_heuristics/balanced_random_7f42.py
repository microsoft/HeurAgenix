from src.problems.max_cut.components import *
import random
from typing import Optional

def balanced_random_7f42(problem_state: dict, algorithm_data: dict, balance_bias: float = 0.7, seed: Optional[int] = None, **kwargs) -> tuple[InsertNodeOperator, dict]:
    """Random node insertion with partition-balance bias. Uniformly samples one unassigned vertex and inserts it into the current smaller set with probability `balance_bias`; otherwise assigns it to a random side (A or B). This promotes balanced partitions during constructive phases without evaluating edge weights, aiding subsequent improvement heuristics that benefit from near-balanced cuts.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "unselected_nodes" (set[int]): The set of vertices not yet assigned to any set.
            - "set_a_count" (int): The number of nodes currently in set A.
            - "set_b_count" (int): The number of nodes currently in set B.
            - "current_solution" (Solution): Not used in this heuristic.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this heuristic.
        balance_bias (float): Probability in [0.0, 1.0] to insert the sampled node into the smaller set (promotes balance). Default is 0.7.
        seed (Optional[int]): Random seed for reproducibility. If provided, RNG will be seeded for this call. Default is None.

    Returns:
        InsertNodeOperator: Operator that inserts the sampled node into set A or set B, biased toward the smaller set by `balance_bias`.
        dict: Empty dictionary as no algorithm data is updated.
    """
    unselected_nodes = problem_state['unselected_nodes']

    # Boundary: if there are no unselected nodes left, return None to signal no applicable move.
    if not unselected_nodes:
        return None, {}

    # Optional reproducibility.
    if seed is not None:
        random.seed(seed)

    # Sample an unassigned node uniformly.
    node_to_insert = random.choice(list(unselected_nodes))

    # Determine smaller set; tie handled by random assignment.
    set_a_count = problem_state['set_a_count']
    set_b_count = problem_state['set_b_count']
    if set_a_count < set_b_count:
        smaller_set = 'A'
    elif set_b_count < set_a_count:
        smaller_set = 'B'
    else:
        smaller_set = None

    # Choose target side with balance bias.
    if smaller_set is not None and random.random() < balance_bias:
        target_set = smaller_set
    else:
        target_set = random.choice(['A', 'B'])

    operator = InsertNodeOperator(node=node_to_insert, target_set=target_set)
    return operator, {}