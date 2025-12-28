from src.problems.max_cut.components import *
import random

def batch_ruin(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[BatchDeleteOperator, dict]:
    """Delete multiple random nodes from the current solution.
    
    Parameters:
    - count (int): Number of nodes to delete. Default is 1.
    - ratio (float): Ratio of nodes to delete (if count is not specified). Default is 0.0.
    """
    current_solution = problem_state.get("current_solution")
    if not current_solution:
        return None, {}
        
    assigned_nodes = list(current_solution.set_a.union(current_solution.set_b))
    if not assigned_nodes:
        return None, {}
    
    count = kwargs.get("count", 0)
    ratio = kwargs.get("ratio", 0.0)
    
    # If count is not provided, calculate from ratio
    if count <= 0 and ratio > 0:
        # Use total nodes in graph or current assigned? 
        # Usually ruin is relative to current solution size.
        count = int(len(assigned_nodes) * ratio)
    
    if count <= 0:
        count = 1
        
    count = min(count, len(assigned_nodes))
    
    nodes_to_delete = random.sample(assigned_nodes, count)
    return BatchDeleteOperator(nodes=nodes_to_delete), {}
