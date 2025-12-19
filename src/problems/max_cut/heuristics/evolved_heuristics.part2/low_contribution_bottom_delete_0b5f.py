from src.problems.max_cut.components import *
import random
import math

def low_contribution_bottom_delete_0b5f(problem_state: dict, algorithm_data: dict, bottom_p: float = 0.5, **kwargs) -> tuple[DeleteOperator, dict]:
    """Delete a node from the bottom fraction of contributors to diversify the partition.
    
    This heuristic identifies vertices with the lowest contribution to the current cut (sum of weights to the opposite set).
    It selects a node uniformly at random from the bottom `bottom_p` fraction of all assigned nodes.
    This introduces significant perturbation by removing nodes that are not strongly anchored to their current set,
    potentially allowing them to be re-inserted into a better position later.
    
    Args:
        problem_state (dict): Contains "weight_matrix" (numpy.ndarray) and "current_solution" (Solution).
        algorithm_data (dict): Not used for input.
        bottom_p (float): The fraction of nodes (0.0 < p <= 1.0) to consider as candidates. Default is 0.5.

    Returns:
        DeleteOperator: Operator to remove the selected node.
        dict: Diagnostics about the selection.
    """
    weight_matrix = problem_state["weight_matrix"]
    current_solution = problem_state["current_solution"]
    set_a = current_solution.set_a
    set_b = current_solution.set_b

    assigned_nodes = list(set_a.union(set_b))
    if not assigned_nodes:
        return None, {}


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

    # Sort by contribution ascending (lowest first)
    contributions.sort(key=lambda x: x[1])

    # Select from bottom p fraction
    p = max(0.0, min(1.0, float(bottom_p)))
    cand_size = max(1, int(math.floor(p * len(contributions)))) if p > 0.0 else 1
    
    candidate_pool = contributions[:cand_size]
    if not candidate_pool:
        return None, {}

    node_to_delete, node_contrib = random.choice(candidate_pool)
    
    return DeleteOperator(node=node_to_delete), {
        "last_deleted_node": node_to_delete,
        "last_deleted_contribution": node_contrib,
        "pool_size": len(candidate_pool)
    }
