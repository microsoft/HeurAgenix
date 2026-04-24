from src.problems.cvrp.components import Solution, SwapStarOperator
import numpy as np

def swap_star(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[SwapStarOperator, dict]:
    distance_matrix = problem_state["distance_matrix"]
    nearest_neighbors = problem_state.get("nearest_neighbors", None)
    depot = problem_state["depot"]
    demands = problem_state["demands"]
    capacity = problem_state["capacity"]
    current_solution = problem_state["current_solution"]

    best_delta = 0
    best_move = None
    
    routes = current_solution.routes
    loads = current_solution.loads
    penalty_factor = problem_state.get("capacity_penalty_factor", 100.0)

    node_to_route_idx = {}
    for r_idx, route in enumerate(routes):
        for idx, node in enumerate(route):
            node_to_route_idx[node] = (r_idx, idx)

    def evaluate_best_insertion(route, node_to_insert, node_to_remove):
        temp_r = [n for n in route if n != node_to_remove]
        n_temp = len(temp_r)
        if n_temp == 0: return 2 * distance_matrix[depot][node_to_insert], 1
        best_add_cost = float('inf')
        best_pos = 1
        for k in range(1, n_temp + 1):
            prev_n = temp_r[k - 1]
            next_n = depot if k == n_temp else temp_r[k]
            add_cost = -distance_matrix[prev_n][next_n] + distance_matrix[prev_n][node_to_insert] + distance_matrix[node_to_insert][next_n]
            if add_cost < best_add_cost:
                best_add_cost = add_cost
                best_pos = k
        return best_add_cost, best_pos

    if nearest_neighbors is not None:
        for node1 in range(1, len(distance_matrix)):
            if node1 not in node_to_route_idx: continue
            r1, idx1 = node_to_route_idx[node1]
            route1 = routes[r1]
            
            for node2 in nearest_neighbors[node1][:20]:
                if node2 not in node_to_route_idx: continue
                r2, idx2 = node_to_route_idx[node2]
                if r1 == r2: continue
                route2 = routes[r2]
                
                prev1 = depot if idx1 == 0 else route1[idx1 - 1]
                next1 = depot if idx1 == len(route1) - 1 else route1[idx1 + 1]
                rem_cost_1 = distance_matrix[prev1][node1] + distance_matrix[node1][next1] - distance_matrix[prev1][next1]
                
                prev2 = depot if idx2 == 0 else route2[idx2 - 1]
                next2 = depot if idx2 == len(route2) - 1 else route2[idx2 + 1]
                rem_cost_2 = distance_matrix[prev2][node2] + distance_matrix[node2][next2] - distance_matrix[prev2][next2]
                
                add_1, pos1 = evaluate_best_insertion(route1, node2, node1)
                if add_1 == float('inf'): continue
                
                add_2, pos2 = evaluate_best_insertion(route2, node1, node2)
                if add_2 == float('inf'): continue
                
                delta_dist = add_1 + add_2 - rem_cost_1 - rem_cost_2
                
                old_p1 = max(0, loads[r1] - capacity)
                old_p2 = max(0, loads[r2] - capacity)
                new_l1 = loads[r1] - demands[node1] + demands[node2]
                new_l2 = loads[r2] - demands[node2] + demands[node1]
                new_p1 = max(0, new_l1 - capacity)
                new_p2 = max(0, new_l2 - capacity)
                delta_pen = (new_p1 + new_p2 - old_p1 - old_p2) * penalty_factor
                
                total_delta = delta_dist + delta_pen
                if total_delta < best_delta - 1e-4:
                    best_delta = total_delta
                    best_move = (r1, r2, node1, node2, pos1, pos2)

    if best_move:
        r1, r2, node1, node2, pos1, pos2 = best_move
        return SwapStarOperator(
            vehicle_id1=r1, node1=node1, best_pos_for_1_in_2=pos2,
            vehicle_id2=r2, node2=node2, best_pos_for_2_in_1=pos1
        ), algorithm_data

    return None, algorithm_data
