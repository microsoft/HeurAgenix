from src.problems.max_cut.components import *
import random
from collections import deque

def batch_cluster_ruin(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[BatchDeleteOperator, dict]:
    """Delete a connected cluster of nodes (Spatial Ruin).
    
    Logic:
    1. Pick a random seed node.
    2. Perform BFS to find its neighbors, and neighbors of neighbors.
    3. Delete this entire connected component.
    
    This is effective for escaping local optima where a whole region is mis-configured.
    """
    current_solution = problem_state.get("current_solution")
    
    # [FIX] instance_data is often flattened into problem_state by env.update_problem_state()
    # So we should look for keys directly, or check instance_data fallback
    adj = problem_state.get("adj")
    if adj is None and "instance_data" in problem_state:
        adj = problem_state["instance_data"].get("adj")
        
    if not current_solution or not adj:
        return None, {}
    
    assigned_nodes = list(current_solution.set_a.union(current_solution.set_b))
    
    if not assigned_nodes:
        return None, {}
    
    count = kwargs.get("count", 0)
    if count <= 0:
        count = max(1, int(len(assigned_nodes) * 0.1))
    count = min(count, len(assigned_nodes))
    
    # 1. Pick random seed
    seed_node = random.choice(assigned_nodes)
    
    # 2. BFS to collect cluster
    cluster = set()
    queue = deque([seed_node])
    cluster.add(seed_node)
    
    # Optimization: Convert assigned_nodes to set for O(1) lookup if needed, 
    # but here we just traverse adj which is implicitly connected.
    # We only care if neighbors are in the current solution.
    
    # To ensure we don't get stuck in a small disconnected component, 
    # we might need multiple seeds.
    
    while len(cluster) < count:
        if not queue:
            # Cluster exhausted but count not reached (disconnected component)
            # Pick a new random seed that is not in cluster
            remaining = [n for n in assigned_nodes if n not in cluster]
            if not remaining:
                break
            new_seed = random.choice(remaining)
            queue.append(new_seed)
            cluster.add(new_seed)
            continue
            
        u = queue.popleft()
        
        neighbors = list(adj[u].keys())
        # Shuffle neighbors to make the shape of cluster random/blob-like
        random.shuffle(neighbors)
        
        for v in neighbors:
            if len(cluster) >= count:
                break
            
            # Only add if it's currently in the solution and not visited
            if v not in cluster and (v in current_solution.set_a or v in current_solution.set_b):
                cluster.add(v)
                queue.append(v)
    
    return BatchDeleteOperator(nodes=list(cluster)), {}
