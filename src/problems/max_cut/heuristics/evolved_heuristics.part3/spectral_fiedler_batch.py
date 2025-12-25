from src.problems.max_cut.components import *
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

def spectral_fiedler_batch(
    problem_state: dict,
    algorithm_data: dict,
    batch_ratio: float = 0.01,
    matrix_choice: str = "laplacian",
    **kwargs
) -> tuple[BatchInsertNodeOperator, dict]:
    """
    Spectral Batch Construction using Fiedler Vector.
    
    Computes the Fiedler vector (second smallest eigenvector of Laplacian) or leading eigenvector
    of the adjacency matrix. Uses the sign of the eigenvector components to determine partition assignment.
    Inserts the top 'batch_ratio' nodes with the highest absolute eigenvector magnitude (confidence).
    
    Mechanism:
    1. Compute eigenvector v.
    2. Sort unselected nodes by |v[i]|.
    3. Select top K nodes.
    4. Assign node i to A if v[i] > 0, else B.
    
    Args:
        batch_ratio (float): Ratio of nodes to insert.
        matrix_choice (str): "laplacian" (default) or "adjacency".
    """
    
    # 1. Check cache
    if "spectral_solution_cache" in algorithm_data:
        solution_cache = algorithm_data["spectral_solution_cache"]
        confidence_scores = algorithm_data["spectral_confidence_scores"]
        unselected_nodes = problem_state["unselected_nodes"]
        
        if not unselected_nodes:
            return None, algorithm_data
            
        node_num = problem_state["node_num"]
        batch_size = max(1, int(node_num * batch_ratio))
        
        nodes_to_a = []
        nodes_to_b = []
        count = 0
        
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
            return None, algorithm_data
            
        return BatchInsertNodeOperator(nodes_to_a=nodes_to_a, nodes_to_b=nodes_to_b), algorithm_data

    # 2. Compute Spectral Solution
    node_num = problem_state["node_num"]
    weight_matrix = problem_state["weight_matrix"]
    
    # Ensure sparse matrix for efficiency
    if not sp.issparse(weight_matrix):
        weight_matrix = sp.csr_matrix(weight_matrix)
        
    # Symmetrize
    W = (weight_matrix + weight_matrix.T) / 2
    
    v = None
    if matrix_choice == "adjacency":
        # Leading eigenvector of Adjacency
        # Use 'LA' (Largest Algebraic)
        vals, vecs = eigsh(W, k=1, which='LA')
        v = vecs[:, 0]
    else:
        # Fiedler vector of Laplacian: L = D - W
        # Smallest non-zero eigenvalue.
        # Construct Laplacian
        degrees = np.array(W.sum(axis=1)).flatten()
        D = sp.diags(degrees)
        L = D - W
        
        # We need the 2nd smallest eigenvector. 
        # 'SM' (Smallest Magnitude) gives smallest eigenvalues.
        # k=2 gives the two smallest. The first is 0 (constant vector), second is Fiedler.
        try:
            # Optimization: Use looser tolerance (0.1) and limit iterations.
            # We only need the sign/direction of the vector, not high precision.
            vals, vecs = eigsh(L, k=2, which='SM', tol=0.1, maxiter=node_num)
            # Sort by eigenvalue just in case
            idx = vals.argsort()
            v = vecs[:, idx[1]] # Second smallest
        except Exception as e:
            # Fallback if eigsh fails (e.g. disconnected graph)
            # Use random vector
            v = np.random.uniform(-1, 1, size=node_num)
            
    # 3. Discretize and Cache
    solution_cache = {}
    confidence_list = []
    
    for i in range(node_num):
        if v[i] > 0:
            solution_cache[i] = 'A'
        else:
            solution_cache[i] = 'B'
        confidence_list.append((i, abs(v[i])))
        
    confidence_list.sort(key=lambda item: item[1], reverse=True)
    
    algorithm_data["spectral_solution_cache"] = solution_cache
    algorithm_data["spectral_confidence_scores"] = confidence_list
    
    # 4. Return first batch
    unselected_nodes = problem_state["unselected_nodes"]
    batch_size = max(1, int(node_num * batch_ratio))
    
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
                
    return BatchInsertNodeOperator(nodes_to_a=nodes_to_a, nodes_to_b=nodes_to_b), algorithm_data
