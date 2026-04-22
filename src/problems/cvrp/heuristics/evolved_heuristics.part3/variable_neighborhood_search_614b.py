from src.problems.cvrp.components import *

def variable_neighborhood_search_614b(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[SwapOperator, dict]:
    current_solution = problem_state.get('current_solution')
    vehicle_loads = problem_state.get('vehicle_loads')
    capacity = problem_state.get('capacity')
    distance_matrix = problem_state.get('distance_matrix')
    demands = problem_state.get('demands')
    nearest_neighbors = problem_state.get('nearest_neighbors')
    depot = problem_state.get('depot', 0)

    best_operator = None
    best_cost_saving = 0.0

    node_to_route = {}
    for vid, route in enumerate(current_solution.routes):
        for pos, node in enumerate(route):
             if node != depot:
                 node_to_route[node] = (vid, pos)

    routes = current_solution.routes

    for nodeA, (vid_A, pos_A) in node_to_route.items():
        if nearest_neighbors is not None:
             candidates = nearest_neighbors[nodeA]
        else:
             candidates = list(node_to_route.keys())

        n_A = len(routes[vid_A])
        prev_A = routes[vid_A][(pos_A - 1) % n_A]
        next_A = routes[vid_A][(pos_A + 1) % n_A]
        demand_A = demands[nodeA]

        for nodeC in candidates:
            if nodeC not in node_to_route:
                 continue
            if nodeC == depot:
                 continue
            
            vid_B, pos_C = node_to_route[nodeC]

            if vid_A == vid_B:
                 continue
            
            demand_C = demands[nodeC]
            
            penalty_factor = problem_state.get('capacity_penalty_factor', 100.0)
            pen1_old = max(0, vehicle_loads[vid_A] - capacity)
            pen2_old = max(0, vehicle_loads[vid_B] - capacity)
            pen1_new = max(0, vehicle_loads[vid_A] - demand_A + demand_C - capacity)
            pen2_new = max(0, vehicle_loads[vid_B] - demand_C + demand_A - capacity)
            penalty_delta = (pen1_new + pen2_new - pen1_old - pen2_old) * penalty_factor

            n_B = len(routes[vid_B])
            prev_C = routes[vid_B][(pos_C - 1) % n_B]
            next_C = routes[vid_B][(pos_C + 1) % n_B]

            cost_increase = -distance_matrix[prev_A][nodeA] - distance_matrix[nodeA][next_A] \
                            -distance_matrix[prev_C][nodeC] - distance_matrix[nodeC][next_C]
            
            cost_increase += distance_matrix[prev_A][nodeC] + distance_matrix[nodeC][next_A] \
                             + distance_matrix[prev_C][nodeA] + distance_matrix[nodeA][next_C]

            cost_increase += penalty_delta
            cost_reduction = -cost_increase

            if cost_reduction > best_cost_saving + 1e-4:
                best_cost_saving = cost_reduction
                best_operator = SwapOperator(vehicle_id1=vid_A, position1=pos_A,
                                            vehicle_id2=vid_B, position2=pos_C)

    if best_operator:
        return best_operator, algorithm_data
    return None, algorithm_data
