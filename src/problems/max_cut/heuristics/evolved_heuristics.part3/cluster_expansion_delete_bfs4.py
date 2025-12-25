from src.problems.max_cut.components import *
import random

def cluster_expansion_delete_bfs4(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[DeleteOperator, dict]:
    """
    Cluster Expansion Deletion (BFS-like).
    
    This heuristic is designed to be called repeatedly to delete a connected cluster of nodes.
    It prioritizes deleting nodes that are neighbors of currently 'unselected' nodes (holes).
    This effectively expands the 'hole' in the solution, creating a large connected void (Cluster Ruin).
    
    If no unselected nodes exist (start of ruin phase) or no neighbors are found, it falls back to random deletion to start a new cluster.
    
    Args:
        problem_state (dict): Contains "current_solution", "unselected_nodes", and "adj" (adjacency list).
        algorithm_data (dict): Not used.
        
    Returns:
        DeleteOperator: Operator to delete a single node that expands the current hole.
    """
    current_solution = problem_state.get("current_solution")
    unselected_nodes = problem_state.get("unselected_nodes")
    adj = problem_state.get("adj") # Use the adjacency list we added to Env
    
    if not current_solution:
        return None, {}

    assigned_nodes = list(current_solution.set_a.union(current_solution.set_b))
    if not assigned_nodes:
        return None, {}

    # If we have unselected nodes (holes), try to expand them
    if unselected_nodes and adj:
        # Find assigned nodes that are neighbors of unselected nodes
        # To be efficient, we can sample a few unselected nodes if there are too many
        
        candidates = []
        
        # Strategy: Sample some unselected nodes and look at their neighbors
        # If unselected_nodes is huge, we don't want to iterate all.
        # But usually in Ruin phase, unselected starts small and grows.
        
        # Let's try to find ANY assigned neighbor of the unselected set.
        # We iterate unselected nodes and collect their assigned neighbors.
        
        # Optimization: If unselected is large, maybe just pick random assigned nodes and check if they have unselected neighbors?
        # No, that's inefficient if the hole is small.
        
        # Let's iterate unselected nodes (limit to first 50 to be fast)
        limit = 50
        count = 0
        for u in unselected_nodes:
            if count > limit:
                break
            if u < len(adj): # Safety check
                for v in adj[u]:
                    if v in current_solution.set_a or v in current_solution.set_b:
                        candidates.append(v)
            count += 1
            
        if candidates:
            # Pick one candidate to delete. 
            # We could pick the one with MOST unselected neighbors to make the hole "round",
            # but random choice is good for stochasticity.
            node_to_delete = random.choice(candidates)
            return DeleteOperator(node=node_to_delete), {}

    # Fallback: Start a new cluster by deleting a random node
    node_to_delete = random.choice(assigned_nodes)
    return DeleteOperator(node=node_to_delete), {}
