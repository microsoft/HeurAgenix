from src.problems.cvrp.components import ReplaceSolutionOperator
import random

def hgs_giant_tour_crossover(problem_state: dict, algorithm_data: dict, target_solution=None, **kwargs) -> tuple[ReplaceSolutionOperator, dict]:
    """
    HGS-inspired Giant Tour Order Crossover (OX) with 2D DP Split.
    It flattens both parents into Giant Tours (TSP), applies OX to inherit chunks of
    consecutive customers from both parents, and then optimally splits the resulting
    Giant Tour into exactly K vehicles using a 2D dynamic programming split.
    """
    parent_a = problem_state["current_solution"]
    depot = problem_state["depot"]
    capacity = problem_state["capacity"]
    demands = problem_state["demands"]
    dist = problem_state["distance_matrix"]
    vehicle_num = problem_state.get("vehicle_num", len(parent_a.routes))
    
    if target_solution is None:
        target_solution = algorithm_data.get("target_solution")
    if target_solution is None:
        return ReplaceSolutionOperator(routes=[list(r) for r in parent_a.routes]), {}

    # Extract giant tours
    giant_tour_a = []
    for route in parent_a.routes:
        for node in route:
            if node != depot:
                giant_tour_a.append(node)
                
    giant_tour_b = []
    for route in target_solution.routes:
        for node in route:
            if node != depot:
                giant_tour_b.append(node)
                
    n = len(giant_tour_a)
    if n == 0 or len(giant_tour_b) != n:
        return ReplaceSolutionOperator(routes=[list(r) for r in parent_a.routes]), {}

    # 1. Order Crossover (OX) on Giant Tours
    # Select two crossover points uniformly at random
    pt1, pt2 = sorted(random.sample(range(n), 2))
    
    # Offspring initialized with None
    offspring_tour = [None] * n
    
    # Copy segment from parent A
    offspring_tour[pt1:pt2+1] = giant_tour_a[pt1:pt2+1]
    
    # Fill the rest with parent B's elements that are not yet in offspring
    in_a_segment = set(offspring_tour[pt1:pt2+1])
    
    # Iterator over B starting from pt2+1
    b_idx = (pt2 + 1) % n
    offspring_idx = (pt2 + 1) % n
    
    while None in offspring_tour:
        candidate = giant_tour_b[b_idx]
        if candidate not in in_a_segment:
            offspring_tour[offspring_idx] = candidate
            offspring_idx = (offspring_idx + 1) % n
        b_idx = (b_idx + 1) % n
        
    # 2. 2D DP Split of offspring_tour into EXACTLY K vehicles
    target_k = vehicle_num
    
    V = [[float('inf')] * (n + 1) for _ in range(target_k + 1)]
    P = [[0] * (n + 1) for _ in range(target_k + 1)]
    
    V[0][0] = 0.0
    penalty_factor = problem_state.get('capacity_penalty_factor', 100.0)
    
    for k in range(1, target_k + 1):
        for i in range(n):
            if V[k-1][i] == float('inf'):
                continue
                
            load = 0
            route_cost = 0.0
            
            for j in range(i + 1, n + 1):
                load += demands[offspring_tour[j - 1]]
                capacity_violation = max(0.0, load - capacity)
                penalty = capacity_violation * penalty_factor
                
                # Single node route
                if j == i + 1:
                    c = offspring_tour[i]
                    route_cost = dist[depot][c] + dist[c][depot]
                else:
                    # Extend route
                    prev_c = offspring_tour[j - 2]
                    curr_c = offspring_tour[j - 1]
                    route_cost += dist[prev_c][curr_c] + dist[curr_c][depot] - dist[prev_c][depot]
                
                total_arc_cost = route_cost + penalty
                
                if V[k-1][i] + total_arc_cost < V[k][j]:
                    V[k][j] = V[k-1][i] + total_arc_cost
                    P[k][j] = i
                    
    # Reconstruct routes
    dp_routes = []
    curr = n
    best_k = target_k
    
    if V[target_k][n] == float('inf'):
        for k in range(target_k, 0, -1):
            if V[k][n] != float('inf'):
                best_k = k
                break
                
    curr_k = best_k
    while curr_k > 0 and curr > 0:
        prev = P[curr_k][curr]
        route = [depot] + offspring_tour[prev:curr]
        dp_routes.append(route)
        curr = prev
        curr_k -= 1
        
    dp_routes.reverse()
    
    while len(dp_routes) < vehicle_num:
        dp_routes.append([depot])
        
    return ReplaceSolutionOperator(routes=dp_routes[:vehicle_num]), {}
