from src.problems.max_cut.components import *
import numpy as np

def weighted_degree_batch(
    problem_state: dict,
    algorithm_data: dict,
    batch_ratio: float = 0.01,
    **kwargs
) -> tuple[BatchInsertNodeOperator, dict]:
    """
    Weighted Degree Batch Construction.
    
    Sorts unselected nodes by their total weighted degree (sum of weights to all other nodes).
    Inserts the top 'batch_ratio' nodes.
    Assignment Strategy: Greedy based on current cut.
    - If a node has more weight to Set B, put in A.
    - If more weight to Set A, put in B.
    - If equal or no connections, put in smaller set (Balance).
    
    This is a batch version of 'most_weight_neighbors'.
    
    Args:
        batch_ratio (float): Ratio of nodes to insert.
    """
    
    weight_matrix = problem_state["weight_matrix"]
    current_solution = problem_state["current_solution"]
    unselected_nodes = problem_state["unselected_nodes"]
    
    if not unselected_nodes:
        return None, {}
        
    # 1. Sort nodes by weighted degree (only computed once and cached)
    if "sorted_nodes_degree" not in algorithm_data:
        # Compute degrees for ALL nodes (static property of graph)
        # Sum of rows.
        # FIX: Use absolute value for sorting to handle negative weights correctly!
        # This ensures nodes with strong connections (positive or negative) are prioritized.
        degrees = np.array(abs(weight_matrix).sum(axis=1)).flatten()
        # Create list of (node, degree)
        all_nodes_sorted = sorted(
            [(i, degrees[i]) for i in range(len(degrees))],
            key=lambda x: x[1],
            reverse=True
        )
        algorithm_data["sorted_nodes_degree"] = all_nodes_sorted
    
    sorted_nodes = algorithm_data["sorted_nodes_degree"]
    
    # 2. Select top K unselected nodes
    node_num = problem_state["node_num"]
    batch_size = max(1, int(node_num * batch_ratio))
    
    nodes_to_insert = []
    count = 0
    
    # Iterate through pre-sorted list to find unselected ones
    for node, _ in sorted_nodes:
        if node in unselected_nodes:
            nodes_to_insert.append(node)
            count += 1
            if count >= batch_size:
                break
                
    # 3. Assign targets
    nodes_to_a = []
    nodes_to_b = []
    
    set_a_list = list(current_solution.set_a)
    set_b_list = list(current_solution.set_b)
    
    # Vectorized gain calculation for the batch
    # We need to know weights from each node in 'nodes_to_insert' to 'set_a' and 'set_b'
    
    if not nodes_to_insert:
        return None, algorithm_data
        
    # Convert to list for indexing
    batch_nodes = nodes_to_insert
    
    # Calculate weights to A and B
    # weight_matrix[batch_nodes][:, set_a_list]
    
    if set_a_list:
        weights_to_a = weight_matrix[batch_nodes][:, set_a_list].sum(axis=1)
        # If result is matrix (sparse), convert to array
        if hasattr(weights_to_a, "toarray"):
            weights_to_a = weights_to_a.toarray().flatten()
        else:
            weights_to_a = np.array(weights_to_a).flatten()
    else:
        weights_to_a = np.zeros(len(batch_nodes))
        
    if set_b_list:
        weights_to_b = weight_matrix[batch_nodes][:, set_b_list].sum(axis=1)
        if hasattr(weights_to_b, "toarray"):
            weights_to_b = weights_to_b.toarray().flatten()
        else:
            weights_to_b = np.array(weights_to_b).flatten()
    else:
        weights_to_b = np.zeros(len(batch_nodes))
        
    # Decide targets
    # Gain to A = weights_to_b
    # Gain to B = weights_to_a
    
    current_a_count = len(set_a_list)
    current_b_count = len(set_b_list)
    
    for i, node in enumerate(batch_nodes):
        gain_if_a = weights_to_b[i]
        gain_if_b = weights_to_a[i]
        
        if gain_if_a > gain_if_b:
            nodes_to_a.append(node)
            current_a_count += 1
        elif gain_if_b > gain_if_a:
            nodes_to_b.append(node)
            current_b_count += 1
        else:
            # Tie: Balance
            if current_a_count <= current_b_count:
                nodes_to_a.append(node)
                current_a_count += 1
            else:
                nodes_to_b.append(node)
                current_b_count += 1
                
    return BatchInsertNodeOperator(nodes_to_a=nodes_to_a, nodes_to_b=nodes_to_b), algorithm_data
