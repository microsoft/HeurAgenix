from src.problems.cvrp.components import *

def or_opt_segment_relocate(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[BaseOperator, dict]:
    """
    Or-opt: Relocate a segment of 1-3 consecutive customers from one route to the best
    position in another route. This is a powerful inter-route operator that moves coherent
    groups, preserving local structure while enabling macro topology changes.
    
    Uses KNN acceleration to limit target route selection.
    Includes soft capacity penalty for infeasible tolerance (HGS-style).
    """
    dist = problem_state["distance_matrix"]
    capacity = problem_state["capacity"]
    depot = problem_state["depot"]
    current_solution = problem_state["current_solution"]
    vehicle_loads = problem_state["vehicle_loads"]
    nearest_neighbors = problem_state.get("nearest_neighbors", None)
    demands = problem_state["demands"]
    penalty_factor = problem_state.get("capacity_penalty_factor", 100.0)

    best_delta = 0.0
    best_op = None
    routes = current_solution.routes

    # Build node -> (vehicle_id, position) map
    node_to_route = {}
    for vid, route in enumerate(routes):
        for pos, node in enumerate(route):
            node_to_route[node] = (vid, pos)

    num_routes = len(routes)
    
    for svid in range(num_routes):
        s_route = routes[svid]
        ns = len(s_route)
        if ns <= 2:  # Only depot + maybe 1 customer, nothing to move
            continue
        
        # Try segments of length 1, 2, 3
        for seg_len in range(1, min(4, ns)):
            for start_pos in range(1, ns - seg_len + 1):  # skip depot at pos 0
                end_pos = start_pos + seg_len - 1  # inclusive
                
                # Segment nodes
                seg_nodes = s_route[start_pos:end_pos + 1]
                seg_demand = sum(demands[n] for n in seg_nodes)
                
                # Cost of removing segment from source route
                prev_s = s_route[start_pos - 1]
                next_s = s_route[end_pos + 1] if end_pos + 1 < ns else s_route[0]  # wrap to depot
                
                # Current edges: prev_s→seg[0], seg[-1]→next_s
                # After removal: prev_s→next_s
                remove_cost = 0.0
                remove_cost -= dist[prev_s][seg_nodes[0]]
                remove_cost -= dist[seg_nodes[-1]][next_s]
                remove_cost += dist[prev_s][next_s]
                # Internal segment edges don't change
                
                # Find target routes via KNN of first segment node
                target_vids = set()
                if nearest_neighbors is not None:
                    for sn in seg_nodes:
                        for nn in nearest_neighbors[sn][:15]:
                            if nn in node_to_route:
                                tvid = node_to_route[nn][0]
                                if tvid != svid:
                                    target_vids.add(tvid)
                else:
                    target_vids = set(range(num_routes)) - {svid}
                
                for tvid in target_vids:
                    t_route = routes[tvid]
                    nt = len(t_route)
                    
                    # Penalty delta
                    old_s_pen = max(0, vehicle_loads[svid] - capacity) * penalty_factor
                    old_t_pen = max(0, vehicle_loads[tvid] - capacity) * penalty_factor
                    new_s_pen = max(0, vehicle_loads[svid] - seg_demand - capacity) * penalty_factor
                    new_t_pen = max(0, vehicle_loads[tvid] + seg_demand - capacity) * penalty_factor
                    pen_delta = (new_s_pen + new_t_pen) - (old_s_pen + old_t_pen)
                    
                    # Find best insertion position in target route
                    for tpos in range(1, nt + 1):
                        prev_t = t_route[tpos - 1]
                        next_t = t_route[tpos] if tpos < nt else t_route[0]  # wrap to depot
                        
                        insert_cost = 0.0
                        insert_cost += dist[prev_t][seg_nodes[0]]
                        insert_cost += dist[seg_nodes[-1]][next_t]
                        insert_cost -= dist[prev_t][next_t]
                        
                        total_delta = remove_cost + insert_cost + pen_delta
                        
                        if total_delta < best_delta - 1e-4:
                            best_delta = total_delta
                            # Use BlockRelocateOperator 
                            best_op = BlockRelocateOperator(
                                source_vehicle_id=svid,
                                start_idx=start_pos,
                                end_idx=end_pos,
                                target_vehicle_id=tvid,
                                target_position=tpos
                            )

    if best_op:
        return best_op, algorithm_data
    return None, algorithm_data
