import numpy as np
import random
import scipy.sparse as sp
from src.problems.max_cut.components import InsertNodeOperator, BatchInsertNodeOperator

def continuous_mean_field_batch(problem_state: dict, algorithm_data: dict, batch_ratio: float = 0.01, **kwargs) -> tuple[BatchInsertNodeOperator, dict]:
    """
    Continuous Mean-Field (CMF) Construction for Max-Cut with Partial Batching.
    
    This version supports 'Hierarchical Hybrid Construction' by allowing partial insertions.
    It calculates the continuous relaxation 'x' for all nodes, but only inserts the top portion
    (defined by 'batch_ratio') of most confident nodes (largest |x|) in each call.
    
    Mechanism:
    1. On the first call, it initializes and evolves a continuous vector 'x'.
    2. It caches the full 'x' vector (confidence scores) and the target assignments.
    3. On each call, it selects the top 'batch_size' (calculated from ratio) unselected nodes.
    4. It returns a BatchInsertNodeOperator for these nodes.
    
    Args:
        batch_ratio (float): Ratio of total nodes to insert in this batch. Default 0.01 (1%).
    """
    
    node_num = problem_state["node_num"]
    # Calculate batch size based on ratio
    batch_size = max(1, int(node_num * batch_ratio))
    
    # 1. Check if we already have a cached solution
    if "cmf_solution_cache" in algorithm_data:
        solution_cache = algorithm_data["cmf_solution_cache"]
        confidence_scores = algorithm_data["cmf_confidence_scores"] # List of (node, |x|)
        unselected_nodes = problem_state["unselected_nodes"]
        
        if not unselected_nodes:
            return None, algorithm_data
            
        # Filter confidence scores to only include currently unselected nodes
        # This is necessary because other heuristics might have inserted some nodes
        # We want the most confident among the *remaining* ones.
        # Optimization: The list is already sorted by confidence. We just iterate and pick.
        
        nodes_to_a = []
        nodes_to_b = []
        count = 0
        
        # We iterate through the pre-sorted confidence list
        # This is O(N) in worst case, but usually we find batch_size nodes quickly.
        # To avoid O(N) every time, we could maintain a pointer or filter the list once.
        # Given N=20k, iterating is fast enough.
        
        for node, _ in confidence_scores:
            if node in unselected_nodes:
                target = solution_cache[node]
                if target == 'A':
                    nodes_to_a.append(node)
                else:
                    nodes_to_b.append(node)
                count += 1
                if count >= batch_size:
                    break
        
        if not nodes_to_a and not nodes_to_b:
             # Fallback
             return None, algorithm_data

        return BatchInsertNodeOperator(nodes_to_a=nodes_to_a, nodes_to_b=nodes_to_b), algorithm_data

    # 2. First Call: Compute the CMF solution
    node_num = problem_state["node_num"]
    weight_matrix = problem_state["weight_matrix"]
    
    # Optimization: Use Sparse Matrix for large graphs
    if node_num > 5000:
        if not sp.issparse(weight_matrix):
            weight_matrix = sp.csr_matrix(weight_matrix)
    
    # Initialize continuous state x
    x = np.random.uniform(-0.1, 0.1, size=node_num)
    
    # Hyperparameters
    iterations = 50
    learning_rate = 0.1 
    
    for _ in range(iterations):
        grad = weight_matrix @ x
        x = x - learning_rate * grad
        
        # Normalize
        max_val = np.max(np.abs(x))
        if max_val > 1e-6:
            x = x / max_val
            
    # 3. Discretize and Cache
    solution_cache = {}
    confidence_list = []
    
    for i in range(node_num):
        if x[i] > 0:
            solution_cache[i] = 'A'
        else:
            solution_cache[i] = 'B'
        confidence_list.append((i, abs(x[i])))
            
    # Sort by confidence (descending)
    confidence_list.sort(key=lambda item: item[1], reverse=True)
    
    # Store in algorithm_data
    algorithm_data["cmf_solution_cache"] = solution_cache
    algorithm_data["cmf_confidence_scores"] = confidence_list
    
    # 4. Return the first batch
    unselected_nodes = problem_state["unselected_nodes"]
    nodes_to_a = []
    nodes_to_b = []
    count = 0
    
    for node, _ in confidence_list:
        if node in unselected_nodes:
            target = solution_cache[node]
            if target == 'A':
                nodes_to_a.append(node)
            else:
                nodes_to_b.append(node)
            count += 1
            if count >= batch_size:
                break
            
    if not nodes_to_a and not nodes_to_b:
        return None, algorithm_data
    
    return BatchInsertNodeOperator(nodes_to_a=nodes_to_a, nodes_to_b=nodes_to_b), algorithm_data
