from src.problems.cvrp.components import BatchRemoveOperator, Solution
import numpy as np
import random
import copy

def guided_capacity_ruin(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[BatchRemoveOperator, dict]:
    capacity = problem_state["capacity"]
    current_solution = problem_state["current_solution"]
    distance_matrix = problem_state["distance_matrix"]
    demands = problem_state["demands"]
    
    num_nodes_to_remove = int(kwargs.get("removal_fraction", 0.15) * problem_state["node_num"])
    num_nodes_to_remove = max(10, min(num_nodes_to_remove, 80))  # bounds
    
    routes = current_solution.routes
    loads = current_solution.loads
    
    # 1. Identify routes that are loosely packed or highly overloaded
    # We want to ruin routes that have either poor packing (< 85% capacity) 
    # or that are severely infeasible.
    route_scores = []
    for idx, (route, load) in enumerate(zip(routes, loads)):
        if len(route) == 0:
            continue
        # Pack ratio: how full is the route?  1.0 is perfect.
        pack_ratio = load / max(1.0, float(capacity))
        
        # If over capacity, heavily penalize. If under capacity, also penalize.
        if pack_ratio > 1.0:
            score = pack_ratio  # High score -> more likely to ruin
        else:
            # Underpack: if pack ratio is 0.8, score is 1.2
            score = 1.0 / max(0.1, pack_ratio)
            
        # Also penalize route length/distance ratio (detours)
        route_dist = sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
        # Add basic depot-to-first and depot-to-last
        depot = problem_state["depot"]
        if len(route) > 0:
            route_dist += distance_matrix[depot, route[0]]
            route_dist += distance_matrix[route[-1], depot]
        
        # We roughly prioritize long distance / low load routes
        avg_dist = route_dist / len(route)
        score *= (avg_dist ** 0.5) 
        
        route_scores.append((score, idx))
        
    if not route_scores:
        return BatchRemoveOperator(nodes=[]), algorithm_data
        
    route_scores.sort(reverse=True, key=lambda x: x[0])
    
    nodes_to_remove = []
    removed_routes = set()
    
    # Randomness: Bias towards the worst routes, but not strict selection
    route_candidates = [idx for _, idx in route_scores[:max(2, len(route_scores)//2)]]
    
    while len(nodes_to_remove) < num_nodes_to_remove and route_candidates:
        r_idx = random.choice(route_candidates)
        route_candidates.remove(r_idx)
        removed_routes.add(r_idx)
        
        route = routes[r_idx]
        for node in route:
             nodes_to_remove.append(node)
             
        # Stop if we have way too many
        if len(nodes_to_remove) > num_nodes_to_remove * 1.5:
            break
            
    # Sometimes take neighbor nodes around the removed routes' centroid to ensure we can re-pack locally
    if len(nodes_to_remove) > 0 and len(nodes_to_remove) < num_nodes_to_remove:
        # compute centroid (by getting a random node that is removed)
        seed_node = random.choice(nodes_to_remove)
        nn = problem_state.get("nearest_neighbors")
        if nn is not None and seed_node < len(nn):
            neighbors = nn[seed_node][:30]
            for nb in neighbors:
                if nn is not None and isinstance(nb, (int, np.integer)) and nb not in nodes_to_remove:
                    nodes_to_remove.append(int(nb))
                    if len(nodes_to_remove) >= num_nodes_to_remove:
                        break

    return BatchRemoveOperator(nodes=nodes_to_remove), algorithm_data
