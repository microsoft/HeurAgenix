from src.problems.cvrp.components import ReplaceSolutionOperator
import itertools

def cross_route_2opt(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple:
    dist = problem_state["distance_matrix"]
    depot = problem_state["depot"]
    demands = problem_state["demands"]
    capacity = problem_state["capacity"]
    penalty_factor = problem_state.get("capacity_penalty_factor", 100.0)
    current_solution = problem_state["current_solution"]

    routes = [list(r) for r in current_solution.routes]
    num_routes = len(routes)
    
    improved = True
    while improved:
        improved = False
        for r1_idx, r2_idx in itertools.combinations(range(num_routes), 2):
            r1 = routes[r1_idx]
            r2 = routes[r2_idx]
            if not r1 or not r2: continue
            
            load1 = sum(demands[n] for n in r1)
            load2 = sum(demands[n] for n in r2)
            
            tail1_demands = [0] * len(r1)
            acc = 0
            for i in range(len(r1)-1, -1, -1):
                acc += demands[r1[i]]
                tail1_demands[i] = acc
                
            tail2_demands = [0] * len(r2)
            acc = 0
            for i in range(len(r2)-1, -1, -1):
                acc += demands[r2[i]]
                tail2_demands[i] = acc
                
            for i in range(-1, len(r1)):
                for j in range(-1, len(r2)):
                    u = r1[i] if i >= 0 else depot
                    next_u = r1[i+1] if i+1 < len(r1) else depot
                    
                    v = r2[j] if j >= 0 else depot
                    next_v = r2[j+1] if j+1 < len(r2) else depot
                    
                    delta_dist = dist[u][next_v] + dist[v][next_u] - dist[u][next_u] - dist[v][next_v]
                    
                    tail1_d = tail1_demands[i+1] if i+1 < len(r1) else 0
                    tail2_d = tail2_demands[j+1] if j+1 < len(r2) else 0
                    
                    new_load1 = load1 - tail1_d + tail2_d
                    new_load2 = load2 - tail2_d + tail1_d
                    
                    old_pen = max(0, load1 - capacity) + max(0, load2 - capacity)
                    new_pen = max(0, new_load1 - capacity) + max(0, new_load2 - capacity)
                    
                    delta_pen = (new_pen - old_pen) * penalty_factor
                    
                    if delta_dist + delta_pen < -1e-4:
                        new_r1 = r1[:i+1] + r2[j+1:]
                        new_r2 = r2[:j+1] + r1[i+1:]
                        if not new_r1 and not new_r2: continue
                        routes[r1_idx] = new_r1
                        routes[r2_idx] = new_r2
                        improved = True
                        break 
                if improved:
                    break
            if improved:
                break
                    
    return ReplaceSolutionOperator(routes=routes), algorithm_data
