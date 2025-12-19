from src.problems.max_cut.components import *
import random

def low_contribution_delete_worst_0b60(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[DeleteOperator, dict]:
    """Delete the single worst contributing node (Greedy Perturbation).
    
    Identifies the node with the absolute lowest contribution to the current cut and removes it.
    This is a deterministic (or near-deterministic if ties exist) pruning operation that removes the 
    "weakest link" in the current solution. It is useful for cleaning up bad moves before rebuilding.
    
    Args:
        problem_state (dict): Contains "weight_matrix" and "current_solution".
        algorithm_data (dict): Not used.

    Returns:
        DeleteOperator: Operator to remove the worst node.
        dict: Diagnostics.
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

    best_node = None
    min_contrib = float('inf')
    candidates = []

    for node in assigned_nodes:
        others = set_b if node in set_a else set_a
        if not others:
            contrib = 0.0
        else:
            contrib = 0.0
            for j in others:
                contrib += undirected_w(node, j)
        
        if contrib < min_contrib:
            min_contrib = contrib
            candidates = [node]
        elif contrib == min_contrib:
            candidates.append(node)

    if not candidates:
        return None, {}

    # Break ties randomly
    node_to_delete = random.choice(candidates)
    
    return DeleteOperator(node=node_to_delete), {
        "last_deleted_node": node_to_delete,
        "last_deleted_contribution": min_contrib
    }
