from src.problems.cvrp.components import InsertOperator

def first_fit_decreasing_bfd(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Bin-Packing First-Fit Decreasing (BFD) Insertion.
    Designed specifically for extremely tight CVRP datasets (like X-series).
    Sorts unvisited nodes by demand size (Largest First), guaranteeing dense packing.
    Then inserts the largest node into the first available vehicle at the min-cost position.
    """
    distance_matrix = problem_state["distance_matrix"]
    capacity = problem_state["capacity"]
    depot = problem_state["depot"]
    current_solution = problem_state["current_solution"]
    vehicle_loads = problem_state["vehicle_loads"]
    unvisited_nodes = problem_state["unvisited_nodes"]
    demands = problem_state["demands"]

    if not unvisited_nodes:
        return None, {}

    # 1. Sort unvisited nodes by Demand descending (Largest items first)
    sorted_unvisited = sorted(unvisited_nodes, key=lambda n: demands[n], reverse=True)
    target_node = sorted_unvisited[0]
    target_demand = demands[target_node]

    best_cost = float('inf')
    b_veh = None
    b_pos = None

    # 2. First Fit: Find vehicles that can legally fit this massive node
    for v_idx, route in enumerate(current_solution.routes):
        if vehicle_loads[v_idx] + target_demand <= capacity:
            # 3. Find the least terrible place to insert it in this specific route
            n = len(route)
            for p in range(1, n + 1):
                prev_n = route[p-1] if p > 0 else depot
                next_n = route[p] if p < n else depot
                cost = distance_matrix[prev_n][target_node] + distance_matrix[target_node][next_n] - distance_matrix[prev_n][next_n]
                
                if cost < best_cost:
                    best_cost = cost
                    b_veh = v_idx
                    b_pos = p

    # 4. Fallback if even the biggest node can't fit ANYWHERE linearly
    if b_veh is None:
        # Find the route with smallest load that can accept it with minimal penalty
        b_veh = min(range(len(vehicle_loads)), key=lambda i: vehicle_loads[i])
        b_pos = max(1, len(current_solution.routes[b_veh]))

    return InsertOperator(b_veh, target_node, b_pos), algorithm_data
