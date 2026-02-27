
import random
from src.problems.max_cut.components import SwapOperator

def batch_flip_perturbation(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[SwapOperator, dict]:
    """
    Batch Flip Perturbation Heuristic.
    
    Selects a random subset of nodes and flips their partition assignment.
    This serves as a generic perturbation/ruin strategy (Light/Medium/Heavy/Noise).
    
    Args:
        problem_state: Standard problem state dictionary containing "node_num".
        algorithm_data: Algorithm context (unused here).
        kwargs:
            ratio (float): Fraction of total nodes to flip (e.g., 0.02 for 2%). 
                           Takes precedence over 'count'.
            count (int): Explicit number of nodes to flip.
            
    Returns:
        SwapOperator: The operator containing the list of nodes to flip.
        dict: Additional info (empty).
    """
    node_num = problem_state["node_num"]
    
    count = kwargs.get("count", 0)
    ratio = kwargs.get("ratio", 0.0)
    
    # Priority: Ratio > Count > Default
    flip_count = 0
    if ratio > 0:
        flip_count = max(1, int(node_num * ratio))
    elif count > 0:
        flip_count = count
        
    if flip_count <= 0:
        # Safety fallback
        return None, {}
        
    # Ensure we don't flip more than exists
    flip_count = min(flip_count, node_num)
    
    # 1. Select Random Nodes
    nodes_to_flip = random.sample(range(node_num), flip_count)
    
    # 2. Return Swap Operator
    # The environment will handle moving A->B and B->A for these nodes.
    return SwapOperator(nodes=nodes_to_flip), {}
