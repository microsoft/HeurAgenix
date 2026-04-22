from src.problems.cvrp.components import Solution, MergeRoutesOperator

def saving_algorithm_710e(problem_state: dict, algorithm_data: dict, merge_threshold: float = 0.0, **kwargs) -> tuple[MergeRoutesOperator, dict]:
    """
    Clarke-Wright savings merge: find the best pair of routes to merge.
    Merges two routes whose endpoints (first/last customer) yield the largest savings.
    Route layout: [depot, c1, c2, ..., cn] with implicit return to depot.
    """
    distance_matrix = problem_state["distance_matrix"]
    vehicle_capacity = problem_state["capacity"]
    depot = problem_state["depot"]
    vehicle_loads = problem_state["vehicle_loads"]
    current_solution = problem_state["current_solution"]
    penalty_factor = problem_state.get('capacity_penalty_factor', 100.0)

    best_saving = merge_threshold
    best_operator = None

    routes = current_solution.routes
    num_routes = len(routes)

    for i in range(num_routes):
        route1 = routes[i]
        if len(route1) <= 1:  # only depot
            continue
        # Last customer in route1 (before implicit return to depot)
        last_i = route1[-1]
        
        for j in range(i + 1, num_routes):
            route2 = routes[j]
            if len(route2) <= 1:  # only depot
                continue
            # First customer in route2 (after depot)
            first_j = route2[1] if len(route2) > 1 else None
            if first_j is None:
                continue
            
            # Savings = d(last_i, depot) + d(depot, first_j) - d(last_i, first_j)
            saving = (distance_matrix[last_i][depot] + distance_matrix[depot][first_j] 
                     - distance_matrix[last_i][first_j])
            
            # Capacity penalty delta
            old_pen_i = max(0, vehicle_loads[i] - vehicle_capacity) * penalty_factor
            old_pen_j = max(0, vehicle_loads[j] - vehicle_capacity) * penalty_factor
            new_pen = max(0, vehicle_loads[i] + vehicle_loads[j] - vehicle_capacity) * penalty_factor
            saving -= (new_pen - old_pen_i - old_pen_j)
            
            if saving > best_saving:
                best_saving = saving
                best_operator = MergeRoutesOperator(source_vehicle_id=j, target_vehicle_id=i)

    if best_operator:
        return best_operator, {}
    return None, {}
