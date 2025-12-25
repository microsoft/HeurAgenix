from src.problems.max_cut.components import *
import random

def random_delete_node_0b5f(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[DeleteOperator, dict]:
    """Delete a completely random node (pure perturbation).
    
    Ignores contribution and simply removes a random node from the current solution.
    Useful for escaping deep local optima where even 'bad' nodes might be part of a good structure.
    """
    current_solution = problem_state.get("current_solution")
    if not current_solution:
        return None, {}
        
    assigned_nodes = list(current_solution.set_a.union(current_solution.set_b))
    if not assigned_nodes:
        return None, {}
        
    node_to_delete = random.choice(assigned_nodes)
    return DeleteOperator(node=node_to_delete), {}
