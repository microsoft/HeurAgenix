from src.problems.max_cut.components import *
import random
import heapq

def batch_worst_ruin(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[BatchDeleteOperator, dict]:
    """Delete the nodes with the worst contribution to the cut value.
    
    Logic:
    1. Calculate the 'cut contribution' of each node (edges to opposite set - edges to same set).
    2. Nodes with low (or negative) contribution are 'unhappy' and should be removed/re-inserted.
    3. This is a greedy ruin strategy.
    """
    current_solution = problem_state.get("current_solution")
    adj = problem_state.get("adj") # Assuming adj is available in problem_state or we need to get it from env
    # Note: problem_state usually contains 'instance_data' which has 'adj'
    
    if not current_solution:
        return None, {}
    
    instance_data = problem_state.get("instance_data")
    if not instance_data:
        return None, {}
        
    adj = instance_data.get("adj")
    
    assigned_nodes = list(current_solution.set_a.union(current_solution.set_b))
    if not assigned_nodes:
        return None, {}
    
    count = kwargs.get("count", 0)
    if count <= 0:
        # Default to 10% if not specified
        count = max(1, int(len(assigned_nodes) * 0.1))
    
    count = min(count, len(assigned_nodes))
    
    # Calculate contribution for a sample of nodes to save time? 
    # Or all nodes? For 20k nodes, O(N*d) is fast enough (d is avg degree).
    
    node_scores = []
    
    # Optimization: Only sample if graph is huge and we only need a few nodes
    # But 'worst' implies we should look at all. Let's look at all.
    
    for u in assigned_nodes:
        contribution = 0
        u_in_a = u in current_solution.set_a
        
        neighbors = adj[u]
        for v, w in neighbors.items():
            if u_in_a:
                if v in current_solution.set_b:
                    contribution += w # Good edge
                elif v in current_solution.set_a:
                    contribution -= w # Bad edge
            else: # u in B
                if v in current_solution.set_a:
                    contribution += w # Good edge
                elif v in current_solution.set_b:
                    contribution -= w # Bad edge
        
        # We want to delete nodes with LOW contribution
        node_scores.append((contribution, u))
    
    # Find the 'count' smallest items
    worst_nodes = heapq.nsmallest(count, node_scores)
    nodes_to_delete = [node for score, node in worst_nodes]
    
    return BatchDeleteOperator(nodes=nodes_to_delete), {}
