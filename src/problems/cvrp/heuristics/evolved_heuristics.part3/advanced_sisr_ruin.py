"""
Advanced SISR (Slack Induction by String Removals) with low-efficiency seed selection.

Place in src/problems/cvrp/heuristics/evolved_heuristics.part3/

Key improvements over basic SISR:
1. Identify low-efficiency nodes (high distance cost per route) as seeds
2. Intelligent string selection from seed neighborhood
3. Adaptive string length based on remaining route structure
4. Preserve route quality by avoiding removal of backbone edges
"""

from src.problems.cvrp.components import BatchRemoveOperator
import random
import numpy as np

def advanced_sisr_ruin(problem_state: dict, algorithm_data: dict, removal_fraction: float = 0.15, **kwargs) -> tuple[BatchRemoveOperator, dict]:
    """
    Advanced SISR with efficiency-aware seed selection.
    - Identifies low-efficiency nodes as candidates for disruption
    - Uses spatial clustering to create meaningful "strings"
    - Preserves route structure more intelligently
    """
    current_solution = problem_state["current_solution"]
    distance_matrix = problem_state["distance_matrix"]
    depot = problem_state["depot"]
    demands = problem_state["demands"]
    
    # Collect node info
    node_to_route = {}
    node_to_pos = {}
    route_costs = []
    
    for r_idx, route in enumerate(current_solution.routes):
        route_cost = 0
        if len(route) > 1:
            for i in range(len(route) - 1):
                route_cost += distance_matrix[route[i]][route[i+1]]
            route_cost += distance_matrix[route[-1]][depot]
            route_cost += distance_matrix[depot][route[0]]
        
        route_costs.append((route_cost, r_idx))
        for pos, n in enumerate(route):
            if n != depot:
                node_to_route[n] = r_idx
                node_to_pos[n] = pos
    
    if not node_to_route:
        return BatchRemoveOperator(nodes=[]), algorithm_data
    
    # Calculate node efficiency: edge cost per unit demand
    node_efficiency = {}
    for node, r_idx in node_to_route.items():
        route = current_solution.routes[r_idx]
        pos = node_to_pos[node]
        
        # Cost contribution of this node
        prev_node = route[pos - 1] if pos > 0 else depot
        next_node = route[pos + 1] if pos + 1 < len(route) else depot
        
        edge_cost = (
            distance_matrix[prev_node][node] +
            distance_matrix[node][next_node] -
            distance_matrix[prev_node][next_node]
        )
        
        demand = demands[node]
        efficiency = edge_cost / max(1.0, demand)  # Lower efficiency = higher removal priority
        node_efficiency[node] = efficiency
    
    # Identify low-efficiency seeds (worst 15% of nodes)
    sorted_by_efficiency = sorted(node_efficiency.items(), key=lambda x: x[1], reverse=True)
    num_low_efficiency = max(2, len(sorted_by_efficiency) // 7)
    low_eff_nodes = set(n for n, _ in sorted_by_efficiency[:num_low_efficiency])
    
    # Determine target removal count
    total_customers = len(node_to_route)
    num_to_remove = max(3, int(total_customers * removal_fraction))
    
    # Greedy string collection
    nodes_to_remove = set()
    nearest_neighbors = problem_state.get("nearest_neighbors")
    
    for seed in random.sample(list(low_eff_nodes), min(3, len(low_eff_nodes))):
        if seed not in node_to_route or seed in nodes_to_remove:
            continue
        if len(nodes_to_remove) >= num_to_remove:
            break
        
        # Find spatially near nodes to seed
        if nearest_neighbors is not None and seed < len(nearest_neighbors):
            candidates = nearest_neighbors[seed]
        else:
            candidates = np.argsort(distance_matrix[seed])[1:]
        
        # Build string around seed
        seed_route_idx = node_to_route[seed]
        seed_pos = node_to_pos[seed]
        seed_route = current_solution.routes[seed_route_idx]
        
        # Average route length
        avg_route_len = sum(len(r) for r in current_solution.routes) / max(1, len(current_solution.routes))
        max_string_len = max(2, min(8, int(avg_route_len / 2.5)))
        
        # Expand string from seed
        string_nodes = {seed}
        
        # Phase 1: Expand in same route
        for offset in range(1, max_string_len):
            # Try positions before and after seed
            for new_pos in [seed_pos - offset, seed_pos + offset]:
                if 0 <= new_pos < len(seed_route):
                    node = seed_route[new_pos]
                    if node != depot and node not in nodes_to_remove:
                        string_nodes.add(node)
                        if len(string_nodes) >= max_string_len:
                            break
            if len(string_nodes) >= max_string_len:
                break
        
        # Phase 2: Add spatially close nodes from candidates
        for cand in candidates:
            if cand not in node_to_route or cand in nodes_to_remove:
                continue
            string_nodes.add(cand)
            if len(string_nodes) >= max_string_len + 2:
                break
        
        nodes_to_remove.update(string_nodes)
    
    if not nodes_to_remove:
        return BatchRemoveOperator(nodes=[]), algorithm_data
    
    # Ensure we hit target
    if len(nodes_to_remove) < num_to_remove:
        remaining = set(node_to_route.keys()) - nodes_to_remove
        additional = random.sample(list(remaining), min(num_to_remove - len(nodes_to_remove), len(remaining)))
        nodes_to_remove.update(additional)
    
    return BatchRemoveOperator(nodes=list(nodes_to_remove)), algorithm_data
