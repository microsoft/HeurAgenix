from src.problems.cvrp.components import *

def three_opt_e8d7(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[ReplaceSolutionOperator, dict]:
    """
    Re-purposed to SWAP* (Best-Insertion Cross-Route Swap) using KNN.
    In SWAP*, we evaluate moving node U from route A and node V from route B.
    Instead of swapping them at their original positions, we insert them into their absolutely best positions 
    in their respective new routes, leading to monumental cost reductions compared to traditional Swap.
    """
    dist = problem_state["distance_matrix"]
    capacity = problem_state["capacity"]
    depot = problem_state["depot"]
    current_solution = problem_state["current_solution"]
    vehicle_loads = problem_state["vehicle_loads"]
    nearest_neighbors = problem_state.get("nearest_neighbors", None)
    demands = problem_state["demands"]

    best_delta = 0.0
    best_move = None

    node_to_route = {}
    for r_idx, route in enumerate(current_solution.routes):
        for pos, node in enumerate(route):
            node_to_route[node] = (r_idx, pos)

    for u, (r_u_idx, pos_u) in node_to_route.items():
        if u == depot: continue
        r_u = current_solution.routes[r_u_idx]
        n_u = len(r_u)
        
        candidates = nearest_neighbors[u] if nearest_neighbors is not None else list(node_to_route.keys())
        
        for v in candidates:
            if v == depot or v not in node_to_route: continue
            r_v_idx, pos_v = node_to_route[v]
            if r_u_idx == r_v_idx: continue
            
            # Penalty evaluation for SWAP*
            new_load1 = vehicle_loads[r_u_idx] - demands[u] + demands[v]
            new_load2 = vehicle_loads[r_v_idx] - demands[v] + demands[u]
            old_1_pen = max(0, vehicle_loads[r_u_idx] - capacity)
            old_2_pen = max(0, vehicle_loads[r_v_idx] - capacity)
            new_1_pen = max(0, new_load1 - capacity)
            new_2_pen = max(0, new_load2 - capacity)
            penalty_delta = (new_1_pen + new_2_pen - old_1_pen - old_2_pen) * 100000.0
            
            r_v = current_solution.routes[r_v_idx]
            n_v = len(r_v)
            
            temp_r_u = r_u[:pos_u] + r_u[pos_u+1:]
            n_tu = len(temp_r_u)
            delta_rem_u = 0.0
            if n_u > 1:
                prev_u = r_u[(pos_u - 1) % n_u]
                next_u = r_u[(pos_u + 1) % n_u]
                delta_rem_u = -dist[prev_u][u] - dist[u][next_u] + dist[prev_u][next_u]

            ins_v_cost = float('inf')
            ins_v_pos = -1
            for i in range(n_tu):
                before = temp_r_u[i]
                after = temp_r_u[(i + 1) % n_tu]
                cost = dist[before][v] + dist[v][after] - dist[before][after]
                if cost < ins_v_cost:
                    ins_v_cost = cost
                    ins_v_pos = i + 1
            if n_tu == 0: ins_v_cost, ins_v_pos = 0.0, 0
            
            temp_r_v = r_v[:pos_v] + r_v[pos_v+1:]
            n_tv = len(temp_r_v)
            delta_rem_v = 0.0
            if n_v > 1:
                prev_v = r_v[(pos_v - 1) % n_v]
                next_v = r_v[(pos_v + 1) % n_v]
                delta_rem_v = -dist[prev_v][v] - dist[v][next_v] + dist[prev_v][next_v]
                
            ins_u_cost = float('inf')
            ins_u_pos = -1
            for i in range(n_tv):
                before = temp_r_v[i]
                after = temp_r_v[(i + 1) % n_tv]
                cost = dist[before][u] + dist[u][after] - dist[before][after]
                if cost < ins_u_cost:
                    ins_u_cost = cost
                    ins_u_pos = i + 1
            if n_tv == 0: ins_u_cost, ins_u_pos = 0.0, 0
                
            total_delta = delta_rem_u + delta_rem_v + ins_v_cost + ins_u_cost
            if total_delta < best_delta - 1e-4:
                best_delta = total_delta
                new_r_u = temp_r_u[:ins_v_pos] + [v] + temp_r_u[ins_v_pos:]
                new_r_v = temp_r_v[:ins_u_pos] + [u] + temp_r_v[ins_u_pos:]
                best_move = (r_u_idx, new_r_u, r_v_idx, new_r_v)
                
    if best_move is not None:
        routes_copy = [list(r) for r in current_solution.routes]
        routes_copy[best_move[0]] = best_move[1]
        routes_copy[best_move[2]] = best_move[3]
        return ReplaceSolutionOperator(routes_copy), algorithm_data
        
    return None, algorithm_data
