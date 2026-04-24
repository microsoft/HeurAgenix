from src.problems.cvrp.components import ReverseSegmentOperator

def two_opt_0554(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[ReverseSegmentOperator, dict]:
    distance_matrix = problem_state["distance_matrix"]
    nearest_neighbors = problem_state.get("nearest_neighbors", None)
    depot = problem_state["depot"]
    current_solution = problem_state["current_solution"]

    best_delta = 0
    best_move = None

    node_to_route_idx = {}
    for route_index, route in enumerate(current_solution.routes):
        for idx, node in enumerate(route):
            node_to_route_idx[node] = (route_index, idx)

    if nearest_neighbors is not None:
        for u in range(1, len(distance_matrix)):
            if u not in node_to_route_idx: continue
            r_idx, i_idx = node_to_route_idx[u]
            route = current_solution.routes[r_idx]
            n = len(route)
            if n <= 2: continue
            
            i = i_idx
            # u is route[i]. Edge is (route[i-1], route[i])
            for v in nearest_neighbors[u][:30]:
                if v not in node_to_route_idx: continue
                v_r_idx, v_idx = node_to_route_idx[v]
                if v_r_idx != r_idx: continue
                
                # j-1 = v_idx -> j = v_idx + 1
                j = v_idx + 1
                
                if i < j:
                    A = route[(i - 1) % n]
                    B = route[i % n]
                    C = route[(j - 1) % n]
                    D = route[j % n]
                    d0 = distance_matrix[A][B] + distance_matrix[C][D]
                    d1 = distance_matrix[A][C] + distance_matrix[B][D]
                    delta = d1 - d0
                    if delta < best_delta - 1e-4:
                        best_delta = delta
                        best_move = (r_idx, [(i, j - 1)])

    if best_move:
        route_index, move_pair = best_move
        return ReverseSegmentOperator(route_index, move_pair), algorithm_data

    return None, algorithm_data
