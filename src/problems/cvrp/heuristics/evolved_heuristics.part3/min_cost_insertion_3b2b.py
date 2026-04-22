from src.problems.cvrp.components import InsertOperator

def min_cost_insertion_3b2b(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Advanced Regret-2 Insertion with Dynamic Capacity Penalty.
    This operator evaluates ALL possible insertions for ALL unvisited nodes,
    adding a penalty if the insertion violates vehicle capacity.
    It guarantees to return an operator if there are unvisited nodes, 
    thus preventing 'Recreate could not complete solution' errors.
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

    # Match the hyper_heuristic/env.py penalty_factor exactly or use default high value
    penalty_factor = problem_state.get("capacity_penalty_factor", 100.0)

    best_node = None
    best_vehicle = None
    best_position = None
    max_regret = -float('inf')
    
    # Track the absolute best insertion across all nodes (in case regret fails to differentiate)
    global_best_cost = float('inf')
    global_b_veh = None
    global_b_pos = None
    global_b_node = None

    for node in unvisited_nodes:
        demand = demands[node]
        best_cost = float('inf')
        second_best_cost = float('inf')
        b_veh = None
        b_pos = None

        for v_idx, route in enumerate(current_solution.routes):
            # Calculate penalty delta (how much NEW penalty this insertion generates)
            current_load = vehicle_loads[v_idx]
            new_load = current_load + demand
            pen_old = max(0, current_load - capacity) * penalty_factor
            pen_new = max(0, new_load - capacity) * penalty_factor
            penalty_increase = pen_new - pen_old

            # Find best insertion slot in this vehicle route
            r_len = len(route)
            if r_len == 0:
                cost = 2 * distance_matrix[depot][node] + penalty_increase
                if cost < best_cost:
                    second_best_cost = best_cost
                    best_cost = cost
                    b_veh = v_idx
                    b_pos = 1  # 1 because depot is handled implicitly
                elif cost < second_best_cost:
                    second_best_cost = cost
                continue

            for p in range(1, r_len + 1):
                prev_n = depot if p == 1 else route[p - 1]
                next_n = depot if p == r_len else route[p]
                
                dist_increase = distance_matrix[prev_n][node] + distance_matrix[node][next_n] - distance_matrix[prev_n][next_n]
                cost = dist_increase + penalty_increase

                if cost < best_cost:
                    second_best_cost = best_cost
                    best_cost = cost
                    b_veh = v_idx
                    b_pos = p
                elif cost < second_best_cost:
                    second_best_cost = cost
                    
        if b_veh is not None:
            # Regret = cost of second best option minus cost of best option
            # If only 1 vehicle exists, second_best is inf, regret is inf -> gets chosen purely on best_cost tiebreakers.
            regret = second_best_cost - best_cost
            if regret > max_regret:
                max_regret = regret
                best_node = node
                best_vehicle = b_veh
                best_position = b_pos
                
            if best_cost < global_best_cost:
                global_best_cost = best_cost
                global_b_veh = b_veh
                global_b_pos = b_pos
                global_b_node = node

    # Fallback to absolute best cost if regret somehow didn't latch
    if best_node is None:
        best_node = global_b_node
        best_vehicle = global_b_veh
        best_position = global_b_pos
        
    return InsertOperator(best_vehicle, best_node, best_position), algorithm_data

