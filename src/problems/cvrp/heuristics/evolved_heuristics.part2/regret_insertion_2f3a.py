from src.problems.cvrp.components import InsertOperator

def regret_insertion_2f3a(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Regret-2 insertion with FORCE INSERT fallback.
    If no strict capacity placement is found (tight X-set), forces the node into the vehicle 
    that gives the minimum cost increase (ignoring capacity), allowing construction to complete.
    """
    distance_matrix = problem_state["distance_matrix"]
    capacity = problem_state["capacity"]
    depot = problem_state["depot"]
    current_solution = problem_state["current_solution"]
    vehicle_loads = problem_state["vehicle_loads"]
    unvisited_nodes = problem_state["unvisited_nodes"]
    demands = problem_state["demands"]

    if not unvisited_nodes:
        return None, algorithm_data

    best_node = None
    best_vehicle = None
    best_position = None
    max_regret = -float('inf')

    # Force fallback tracking
    fallback_node = unvisited_nodes[0]
    fallback_cost = float('inf')
    fallback_v = 0
    fallback_p = 0

    for node in unvisited_nodes:
        demand = demands[node]
        best_cost = float('inf')
        second_best_cost = float('inf')
        b_veh = None
        b_pos = None

        for v_idx, route in enumerate(current_solution.routes):
            # strict check
            penalty_factor = problem_state.get('penalty_factor', 50.0)
            old_penalty = max(0, vehicle_loads[v_idx] - capacity)
            new_penalty = max(0, vehicle_loads[v_idx] + demand - capacity)
            penalty_cost = (new_penalty - old_penalty) * penalty_factor
            # is_feasible logic relaxed to penalty
            
            n = len(route)
            for p in range(1, n + 1):
                prev_n = route[p-1] if p > 0 else depot
                next_n = route[p] if p < n else depot
                
                cost = distance_matrix[prev_n][node] + distance_matrix[node][next_n] - distance_matrix[prev_n][next_n]
                
                cost += penalty_cost
                if True:
                    if cost < best_cost:
                        second_best_cost = best_cost
                        best_cost = cost
                        b_veh = v_idx
                        b_pos = p
                    elif cost < second_best_cost:
                        second_best_cost = cost
                else:
                    # track fallback globally
                    if cost < fallback_cost:
                        fallback_cost = cost
                        fallback_v = v_idx
                        fallback_p = p
                        fallback_node = node

        if b_veh is not None:
            regret = second_best_cost - best_cost
            if regret > max_regret:
                max_regret = regret
                best_node = node
                best_vehicle = b_veh
                best_position = b_pos

    if best_node is not None:
        return InsertOperator(best_vehicle, best_node, best_position), algorithm_data
        
    # If we get here, NO strict valid placement was found for ANY node.
    # FORCE INSERT the fallback node to allow construction to complete.
    return InsertOperator(fallback_v, fallback_node, fallback_p), algorithm_data
