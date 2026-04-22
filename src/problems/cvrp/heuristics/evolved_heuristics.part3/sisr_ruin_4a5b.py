from src.problems.cvrp.components import BatchRemoveOperator
import random
import numpy as np

def sisr_ruin_4a5b(problem_state: dict, algorithm_data: dict, removal_fraction: float = 0.15, **kwargs) -> tuple[BatchRemoveOperator, dict]:
    """
    Slack Induction by String Removals (SISRs) - A SOTA Ruin heuristic for highly constrained CVRPs.
    Proposed by Christiaens & Vanden Berghe (2020).
    Instead of randomly removing nodes, it selects a spatial seed and removes contiguous "strings" 
    of nodes from routes that pass near the seed. This preserves some route structure while 
    creating concentrated "slack" (capacity and space) allowing Recreate to find radically different topologies.
    """
    current_solution = problem_state["current_solution"]
    distance_matrix = problem_state["distance_matrix"]
    depot = problem_state["depot"]
    
    total_customers = sum(len(route) for route in current_solution.routes) - current_solution.routes.count([depot])  # rough approx
    if total_customers <= 0:
        return None, algorithm_data
        
    num_to_remove = max(4, int(problem_state["node_num"] * removal_fraction))
    
    # 1. Pick a random seed node that is currently in a route
    valid_nodes = []
    node_to_route_idx = {}
    for r_idx, route in enumerate(current_solution.routes):
        for pos, n in enumerate(route):
            if n != depot:
                valid_nodes.append(n)
                node_to_route_idx[n] = (r_idx, pos)
                
    if not valid_nodes:
        return None, algorithm_data
        
    seed = random.choice(valid_nodes)
    
    # 2. Find nodes closest to the seed
    # Use precalculated nearest_neighbors if available
    nearest_neighbors = problem_state.get("nearest_neighbors", None)
    if nearest_neighbors is not None:
        closest = nearest_neighbors[seed]
    else:
        closest = np.argsort(distance_matrix[seed])[1:] # Exclude seed itself
        
    # Keep adding strings near the seed until we hit the quota
    nodes_to_remove = set()
    
    # Max string length (L_max). Typical bounded by 1/6 of average route length or 10
    avg_route_len = total_customers / max(1, len(current_solution.routes))
    l_max = max(2, min(10, int(avg_route_len / 2)))
    
    # Prioritize strings surrounding the seed's neighborhood
    for candidate in [seed] + list(closest):
        if len(nodes_to_remove) >= num_to_remove:
            break
            
        if candidate not in node_to_route_idx or candidate in nodes_to_remove:
            continue
            
        r_idx, pos = node_to_route_idx[candidate]
        route = current_solution.routes[r_idx]
        
        # Determine a random string length from 1 to l_max
        string_len = random.randint(1, l_max)
        
        # Determine segment around `pos`
        # To avoid wrapping around the depot, clamp the indices
        start_pos = max(0, pos - string_len // 2)
        end_pos = min(len(route), start_pos + string_len)
        
        # Extract the string of nodes
        for idx in range(start_pos, end_pos):
            node_in_string = route[idx]
            if node_in_string != depot:
                nodes_to_remove.add(node_in_string)
                
    if not nodes_to_remove:
        return None, algorithm_data
        
    return BatchRemoveOperator(list(nodes_to_remove)), algorithm_data
