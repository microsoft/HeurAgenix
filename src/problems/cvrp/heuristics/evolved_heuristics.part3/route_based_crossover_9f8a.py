from src.problems.cvrp.components import ReplaceSolutionOperator
import random

def route_based_crossover_9f8a(problem_state: dict, algorithm_data: dict, target_solution=None, **kwargs) -> tuple[ReplaceSolutionOperator, dict]:
    parent_a = problem_state["current_solution"]
    depot = problem_state["depot"]
    capacity = problem_state["capacity"]
    demands = problem_state["demands"]
    dist = problem_state["distance_matrix"]
    
    if target_solution is None:
        target_solution = algorithm_data.get("target_solution")
    if target_solution is None:
        return ReplaceSolutionOperator(routes=[list(r) for r in parent_a.routes]), {}

    valid_routes_a = [r for r in parent_a.routes if len([n for n in r if n != depot]) > 0]
    if not valid_routes_a:
        return ReplaceSolutionOperator(routes=[list(r) for r in target_solution.routes]), {}
        
    chosen_route_a = random.choice(valid_routes_a)
    nodes_in_a = set(n for n in chosen_route_a if n != depot)
    
    remaining_giant_tour = []
    for route_b in target_solution.routes:
        for node in route_b:
            if node != depot and node not in nodes_in_a:
                remaining_giant_tour.append(node)
                
    vehicle_num = problem_state.get("vehicle_num", len(parent_a.routes))
    target_k = vehicle_num - 1
    
    if not remaining_giant_tour:
        offspring_routes = [list(chosen_route_a)]
        while len(offspring_routes) < vehicle_num:
            offspring_routes.append([depot])
        return ReplaceSolutionOperator(routes=offspring_routes[:vehicle_num]), {}
        
    n = len(remaining_giant_tour)
    
    # V[k][i] = min cost to serve first i customers with k routes
    # Initialize with infinity
    V = [[float('inf')] * (n + 1) for _ in range(target_k + 1)]
    P = [[0] * (n + 1) for _ in range(target_k + 1)]
    
    V[0][0] = 0.0
    
    # 2D DP Split
    for k in range(1, target_k + 1):
        for i in range(n):
            if V[k-1][i] == float('inf'):
                continue
                
            load = 0
            route_cost = 0.0
            
            for j in range(i + 1, n + 1):
                load += demands[remaining_giant_tour[j - 1]]
                capacity_violation = max(0.0, load - capacity)
                penalty = capacity_violation * problem_state.get('capacity_penalty_factor', 100.0)
                
                if j == i + 1:
                    c = remaining_giant_tour[i]
                    route_cost = dist[depot][c] + dist[c][depot]
                else:
                    prev_c = remaining_giant_tour[j - 2]
                    curr_c = remaining_giant_tour[j - 1]
                    route_cost += dist[prev_c][curr_c] + dist[curr_c][depot] - dist[prev_c][depot]
                
                total_arc_cost = route_cost + penalty
                
                if V[k-1][i] + total_arc_cost < V[k][j]:
                    V[k][j] = V[k-1][i] + total_arc_cost
                    P[k][j] = i
                    
    # Reconstruct routes
    # Start from V[target_k][n]
    dp_routes = []
    curr = n
    best_k = target_k
    
    # If somehow strict target_k failed to reach n, find the max k that reached n (should not happen because of penalty)
    if V[target_k][n] == float('inf'):
        for k in range(target_k, 0, -1):
            if V[k][n] != float('inf'):
                best_k = k
                break
                
    curr_k = best_k
    while curr_k > 0 and curr > 0:
        prev = P[curr_k][curr]
        route = [depot] + remaining_giant_tour[prev:curr]
        dp_routes.append(route)
        curr = prev
        curr_k -= 1
        
    dp_routes.reverse()
    
    offspring_routes = [list(chosen_route_a)] + dp_routes
    while len(offspring_routes) < vehicle_num:
        offspring_routes.append([depot])
        
    return ReplaceSolutionOperator(routes=offspring_routes[:vehicle_num]), {}
