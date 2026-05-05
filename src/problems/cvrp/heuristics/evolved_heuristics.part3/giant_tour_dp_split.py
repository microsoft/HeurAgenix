from src.problems.cvrp.components import ReplaceSolutionOperator

def giant_tour_dp_split(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[ReplaceSolutionOperator, dict]:
    dist = problem_state["distance_matrix"]
    capacity = problem_state["capacity"]
    depot = problem_state["depot"]
    current_solution = problem_state["current_solution"]
    demands = problem_state["demands"]
    vehicle_num = len(current_solution.routes)
    penalty_factor = problem_state.get('capacity_penalty_factor', 100.0)

    giant_tour = []
    for route in current_solution.routes:
        for node in route:
            if node != depot:
                giant_tour.append(node)

    n = len(giant_tour)
    if n == 0:
        return None, algorithm_data

    V = [[float('inf')] * (n + 1) for _ in range(vehicle_num + 1)]
    V[0][0] = 0.0
    
    P = [[0] * (n + 1) for _ in range(vehicle_num + 1)]

    for k in range(1, vehicle_num + 1):
        for i in range(k - 1, n):
            if V[k - 1][i] == float('inf'): continue
            load = 0
            route_cost = 0.0
            for j in range(i + 1, n + 1):
                load += demands[giant_tour[j - 1]]
                penalty = max(0, load - capacity) * penalty_factor
                if j == i + 1:
                    c = giant_tour[i]
                    route_cost = dist[depot][c] + dist[c][depot]
                else:
                    prev_c = giant_tour[j - 2]
                    curr_c = giant_tour[j - 1]
                    route_cost += dist[prev_c][curr_c] + dist[curr_c][depot] - dist[prev_c][depot]
                total_cost = V[k - 1][i] + route_cost + penalty
                if total_cost < V[k][j]:
                    V[k][j] = total_cost
                    P[k][j] = i

    best_k = -1
    best_cost = float('inf')
    for k in range(1, vehicle_num + 1):
        if V[k][n] < best_cost:
            best_cost = V[k][n]
            best_k = k

    if best_cost == float('inf'):
        return None, algorithm_data

    routes = []
    curr = n
    curr_k = best_k
    while curr > 0 and curr_k > 0:
        prev = P[curr_k][curr]
        routes.append([depot] + giant_tour[prev:curr])
        curr = prev
        curr_k -= 1
        
    routes.reverse()
    while len(routes) < vehicle_num:
        routes.append([depot])
        
    old_total_cost = 0.0
    for r in current_solution.routes:
        if not r: continue
        rc = dist[depot][r[0]] + sum(dist[r[i]][r[i+1]] for i in range(len(r)-1)) + dist[r[-1]][depot]
        old_total_cost += rc
    old_total_cost += sum((max(0, sum(demands[n] for n in r) - capacity) * penalty_factor) for r in current_solution.routes)
    
    if best_cost < old_total_cost - 1e-4:
        return ReplaceSolutionOperator(routes=routes), algorithm_data
    return None, algorithm_data
