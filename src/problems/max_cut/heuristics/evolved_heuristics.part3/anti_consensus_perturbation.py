
import random
from src.problems.max_cut.components import SwapOperator

def anti_consensus_perturbation(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[SwapOperator, dict]:
    """
    Anti-Consensus Perturbation (Supernova).
    
    Identifies 'static' nodes (consensus) across the elite pool and forcibly flips them.
    This is designed to break out of deep convergence basins where the entire population 
    has agreed on a suboptimal partition for certain nodes.
    
    Args:
        problem_state: standard problem state
        algorithm_data: must contain "elite_pool"
        kwargs: 
            ratio (float): Fraction of static nodes to flip (default 0.3-0.5)
            
    Returns:
        SwapOperator: containing the nodes to flip
    """
    elite_pool = algorithm_data.get("elite_pool", [])
    if not elite_pool:
        return None, {}

    node_num = problem_state["node_num"]
    pool_size = len(elite_pool)
    
    # 1. Identify Consensus Nodes
    # Count occurrence of each node in set_a
    set_a_counts = {}
    for sol in elite_pool:
        for node in sol.set_a:
            set_a_counts[node] = set_a_counts.get(node, 0) + 1
            
    static_nodes = []
    for node in range(node_num):
        count = set_a_counts.get(node, 0)
        # Static if always in A (count == size) or always in B (count == 0)
        if count == pool_size or count == 0:
            static_nodes.append(node)
            
    if not static_nodes:
        # Fallback if no absolute consensus (rare)
        return None, {}

    # 2. Select Subset to Flip
    ratio = kwargs.get("ratio", 0.3)
    flip_count = max(1, int(len(static_nodes) * ratio))
    
    nodes_to_flip = random.sample(static_nodes, flip_count)
    
    # 3. Return Operator
    # SwapOperator handles moving nodes from A->B or B->A automatically
    op = SwapOperator(nodes=nodes_to_flip)
    
    return op, {}
