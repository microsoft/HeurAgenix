from src.problems.cvrp.components import *

def node_shift_between_routes_7b8a(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[RelocateOperator, dict]:
    """
    Fast Inter-route Relocate (KNN-accelerated).
    Evaluates relocating a node from its current route to another route, but restricts target routes 
    to those containing at least one K-nearest neighbor of the shifted node, dropping O(N^2) load.
    """
    dist = problem_state["distance_matrix"]
    capacity = problem_state["capacity"]
    depot = problem_state["depot"]
    current_solution = problem_state["current_solution"]
    vehicle_loads = problem_state["vehicle_loads"]
    nearest_neighbors = problem_state.get("nearest_neighbors", None)
    demands = problem_state["demands"]

    best_cost_reduction = 0.0
    best_operator = None

    node_to_route = {}
    for vid, route in enumerate(current_solution.routes):
        for pos, node in enumerate(route):
            node_to_route[node] = (vid, pos)

    for u, (svid, spos) in node_to_route.items():
        if u == depot: continue
        if spos == 0: continue  # depot position
        demand_u = demands[u]

        candidates = nearest_neighbors[u] if nearest_neighbors is not None else list(node_to_route.keys())
        
        target_routes_to_check = set()
        for v in candidates:
            if v in node_to_route:
                tvid = node_to_route[v][0]
                if tvid != svid:
                    target_routes_to_check.add(tvid)
        
        # Also add a few random routes to avoid missing improvements outside KNN
        if nearest_neighbors is not None and len(current_solution.routes) > 2:
            all_vids = [vid for vid in range(len(current_solution.routes)) if vid != svid and len(current_solution.routes[vid]) > 1]
            import random
            for vid in random.sample(all_vids, min(3, len(all_vids))):
                target_routes_to_check.add(vid)
                    
        for tvid in target_routes_to_check:
            # Use soft capacity penalties instead of hard rejection.
            penalty_factor = problem_state.get('capacity_penalty_factor', 100.0)
            old_s_penalty = max(0, vehicle_loads[svid] - capacity) * penalty_factor
            old_t_penalty = max(0, vehicle_loads[tvid] - capacity) * penalty_factor
            new_s_penalty = max(0, vehicle_loads[svid] - demand_u - capacity) * penalty_factor
            new_t_penalty = max(0, vehicle_loads[tvid] + demand_u - capacity) * penalty_factor
            penalty_delta = new_s_penalty + new_t_penalty - old_s_penalty - old_t_penalty

            
            s_route = current_solution.routes[svid]
            t_route = current_solution.routes[tvid]
            ns = len(s_route)
            nt = len(t_route)
            
            # Source: removal cost (CVRP circular: depot=route[0], last node wraps to depot)
            prev_s = s_route[spos - 1] if spos > 0 else depot
            next_s = s_route[spos + 1] if spos < ns - 1 else s_route[0]  # wrap to depot
            cost_rem = -dist[prev_s][u] - dist[u][next_s] + dist[prev_s][next_s]
            
            # Target: try inserting after each position (skip position 0 = depot) 
            for tpos in range(1, nt + 1):
                prev_t = t_route[tpos - 1]
                next_t = t_route[tpos] if tpos < nt else t_route[0]  # wrap to depot
                
                cost_add = dist[prev_t][u] + dist[u][next_t] - dist[prev_t][next_t]
                
                cost_reduction = -(cost_rem + cost_add + penalty_delta)
                if cost_reduction > best_cost_reduction + 1e-4:
                    best_cost_reduction = cost_reduction
                    best_operator = RelocateOperator(
                        source_vehicle_id=svid,
                        source_position=spos,
                        target_vehicle_id=tvid,
                        target_position=tpos
                    )

    if best_operator:
        return best_operator, algorithm_data
    return None, algorithm_data
