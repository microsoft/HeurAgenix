from src.problems.max_cut.components import *
import random
from typing import Optional

def balanced_random_batch(problem_state: dict, algorithm_data: dict, batch_ratio: float = 0.01, balance_bias: float = 0.7, seed: Optional[int] = None, **kwargs) -> tuple[BatchInsertNodeOperator, dict]:
    """
    Random node insertion with partition-balance bias (Batch Version).
    
    Selects a portion of unassigned vertices (defined by 'batch_ratio') and inserts them.
    It tries to maintain balance by assigning nodes to the smaller set with probability `balance_bias`.
    
    Args:
        batch_ratio (float): Ratio of total nodes to insert in this batch. Default 0.01.
        balance_bias (float): Probability to insert into the smaller set.
    """
    unselected_nodes = list(problem_state['unselected_nodes'])

    if not unselected_nodes:
        return None, {}

    if seed is not None:
        random.seed(seed)

    # Determine how many to pick
    node_num = problem_state.get("node_num", len(unselected_nodes) + problem_state.get("set_a_count", 0) + problem_state.get("set_b_count", 0))
    batch_size = max(1, int(node_num * batch_ratio))
    
    k = min(len(unselected_nodes), batch_size)
    nodes_to_insert = random.sample(unselected_nodes, k)
    
    nodes_to_a = []
    nodes_to_b = []
    
    # Current counts
    set_a_count = problem_state['set_a_count']
    set_b_count = problem_state['set_b_count']
    
    for node in nodes_to_insert:
        # Re-evaluate smaller set dynamically or statically?
        # Statically for the batch is faster and fine for approximate balance.
        
        if set_a_count < set_b_count:
            smaller_set = 'A'
        elif set_b_count < set_a_count:
            smaller_set = 'B'
        else:
            smaller_set = None
            
        if smaller_set is not None and random.random() < balance_bias:
            target = smaller_set
        else:
            target = random.choice(['A', 'B'])
            
        if target == 'A':
            nodes_to_a.append(node)
            set_a_count += 1
        else:
            nodes_to_b.append(node)
            set_b_count += 1
            
    return BatchInsertNodeOperator(nodes_to_a=nodes_to_a, nodes_to_b=nodes_to_b), {}
