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
        demand_u = demands[u]

        candidates = nearest_neighbors[u] if nearest_neighbors is not None else list(node_to_route.keys())
        
        target_routes_to_check = set()
        for v in candidates:
            if v in node_to_route:
                tvid = node_to_route[v][0]
                if tvid != svid:
                    target_routes_to_check.add(tvid)
                    
        for tvid in target_routes_to_check:
            # Penalty evaluation for relocate
            old_s_penalty = max(0, vehicle_loads[svid] - capacity)
            old_t_penalty = max(0, vehicle_loads[tvid] - capacity)
            new_s_penalty = max(0, vehicle_loads[svid] - demand_u - capacity)
            new_t_penalty = max(0, vehicle_loads[tvid] + demand_u - capacity)
            penalty_factor = 100000.0
            penalty_delta = (new_s_penalty + new_t_penalty - old_s_penalty - old_t_penalty) * penalty_factor

            
            s_route = current_solution.routes[svid]
            t_route = current_solution.routes[tvid]
            ns = len(s_route)
            nt = len(t_route)
            
            prev_s = s_route[(spos - 1) % ns]
            next_s = s_route[(spos + 1) % ns]
            cost_rem = -dist[prev_s][u] - dist[u][next_s] + dist[prev_s][next_s]
            
            for tpos in range(1, nt + 1):
                prev_t = t_route[(tpos - 1) % nt]
                next_t = t_route[tpos % nt]
                
                cost_add = dist[prev_t][u] + dist[u][next_t] - dist[prev_t][next_t]
                
                cost_reduction = -(cost_rem + cost_add)
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
