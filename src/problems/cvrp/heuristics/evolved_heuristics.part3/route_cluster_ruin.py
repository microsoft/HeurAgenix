from src.problems.cvrp.components import BatchRemoveOperator, Solution
import numpy as np
import random

def route_cluster_ruin(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[BatchRemoveOperator, dict]:
    """
    Destroys a geographically dense cluster of entire routes.
    Picks a seed route, finds the N closest routes to it, and completely disintegrates all of them.
    """
    current_solution = problem_state["current_solution"]
    distance_matrix = problem_state["distance_matrix"]
    
    # Base configuration: fraction of total nodes to remove
    removal_fraction = kwargs.get("removal_fraction", 0.15)
    target_nodes_to_remove = int(problem_state["node_num"] * removal_fraction)
    target_nodes_to_remove = max(10, min(target_nodes_to_remove, int(problem_state["node_num"] * 0.3)))

    routes = current_solution.routes
    valid_routes = [r for r in routes if len(r) > 0]
    
    if not valid_routes:
        return BatchRemoveOperator(nodes=[]), algorithm_data

    # 1. Pick a seed route randomly, weighted slightly towards smaller/worse routes
    seed_route_idx = random.randint(0, len(valid_routes) - 1)
    seed_route = valid_routes[seed_route_idx]
    
    # 2. Find a representative node for the seed route (e.g., its median node or just a random one)
    seed_node = random.choice(seed_route)
    
    # 3. Calculate distance from seed_node to all other routes
    route_distances = []
    for i, r in enumerate(valid_routes):
        if i == seed_route_idx:
            route_distances.append((0, i))
            continue
            
        # Distance to route = min distance to any node in the route
        min_dist = min(distance_matrix[seed_node][n] for n in r)
        route_distances.append((min_dist, i))
        
    # Sort routes by proximity to the seed route
    route_distances.sort(key=lambda x: x[0])
    
    # 4. Greedily collect entire routes until we meet/exceed our removal quota
    nodes_to_remove = []
    routes_destroyed = 0
    
    for dist, r_idx in route_distances:
        target_route = valid_routes[r_idx]
        nodes_to_remove.extend(target_route)
        routes_destroyed += 1
        
        # Add some jitter to the stopping condition
        if len(nodes_to_remove) >= target_nodes_to_remove:
            # 50% chance to drag in one more route to ensure massive structural break
            if random.random() > 0.5:
                break

    # 5. Remove depot just in case
    depot = problem_state["depot"]
    nodes_to_remove = list(set([n for n in nodes_to_remove if n != depot]))

    return BatchRemoveOperator(nodes=nodes_to_remove), algorithm_data
