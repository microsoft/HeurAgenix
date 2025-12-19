from src.problems.max_cut.components import *
import random

def low_contribution_delete_worst_3_0b63(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[DeleteOperator, dict]:
    """Delete one of the 3 worst contributing nodes.
    
    Selects the 3 nodes with the lowest contributions to the cut and removes one at random.
    This balances strict greedy pruning (worst-1) with a small amount of stochasticity,
    preventing the algorithm from always deleting the exact same node in similar situations.
    
    Args:
        problem_state (dict): Contains "weight_matrix" and "current_solution".
        algorithm_data (dict): Not used.

    Returns:
        DeleteOperator: Operator to remove one of the worst 3 nodes.
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

    # Sort by contribution ascending
    contributions.sort(key=lambda x: x[1])

    # Pick from top 3 worst (lowest contribution)
    k = 3
    candidate_pool = contributions[:k]
    
    if not candidate_pool:
        return None, {}

    node_to_delete, node_contrib = random.choice(candidate_pool)
    
    return DeleteOperator(node=node_to_delete), {
        "last_deleted_node": node_to_delete,
        "last_deleted_contribution": node_contrib,
        "pool_size": len(candidate_pool)
    }
