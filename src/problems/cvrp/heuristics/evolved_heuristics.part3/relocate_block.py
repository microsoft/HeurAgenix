from src.problems.cvrp.components import ReplaceSolutionOperator
import itertools

def relocate_block(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple:
    dist = problem_state["distance_matrix"]
    depot = problem_state["depot"]
    demands = problem_state["demands"]
    capacity = problem_state["capacity"]
    penalty_factor = problem_state.get("capacity_penalty_factor", 100.0)
    current_solution = problem_state["current_solution"]

    if "knn_20" not in algorithm_data:
        num_nodes = len(dist)
        K = min(20, num_nodes)
        knn = []
        for i in range(num_nodes):
            neighbors = sorted(range(num_nodes), key=lambda x: dist[i][x])
            knn.append(set(neighbors[1:K+1]))
        algorithm_data["knn_20"] = knn
    knn = algorithm_data["knn_20"]

    routes = [list(r) for r in current_solution.routes]
    
    improved = True
    while improved:
        improved = False
        num_routes = len(routes)
        for r1_idx, r2_idx in itertools.permutations(range(num_routes), 2):
            r1 = routes[r1_idx]
            r2 = routes[r2_idx]
            if not r1: continue
            
            load1 = sum(demands[n] for n in r1)
            load2 = sum(demands[n] for n in r2)
            
            for L in [1, 2, 3]: # Block lengths 1 to 3
                if len(r1) < L: continue
                
                for i in range(len(r1) - L + 1):
                    block = r1[i:i+L]
                    block_load = sum(demands[n] for n in block)
                    
                    new_load1 = load1 - block_load
                    new_load2 = load2 + block_load
                    
                    old_pen = max(0, load1 - capacity) + max(0, load2 - capacity)
                    new_pen = max(0, new_load1 - capacity) + max(0, new_load2 - capacity)
                    delta_pen = (new_pen - old_pen) * penalty_factor
                    
                    # Nodes before and after the block in route 1
                    u = r1[i-1] if i > 0 else depot
                    v = r1[i+L] if i+L < len(r1) else depot
                    
                    block_start = block[0]
                    block_end = block[-1]
                    
                    # Distance change in route 1
                    dist_rem_r1 = dist[u][block_start] + dist[block_end][v] - dist[u][v]
                    
                    for j in range(-1, len(r2)):
                        # Nodes before and after insertion point in route 2
                        x = r2[j] if j >= 0 else depot
                        y = r2[j+1] if j+1 < len(r2) else depot
                        
                        if block_start not in knn[x] and block_end not in knn[y]:
                            continue
                        
                        # Distance change in route 2
                        dist_add_r2 = dist[x][block_start] + dist[block_end][y] - dist[x][y]
                        
                        delta_dist = dist_add_r2 - dist_rem_r1
                        
                        # Greedy acceptance strictly improving cost
                        if delta_dist + delta_pen < -1e-4:
                            new_r2 = r2[:j+1] + block + r2[j+1:]
                            new_r1 = r1[:i] + r1[i+L:]
                            routes[r1_idx] = new_r1
                            routes[r2_idx] = new_r2
                            improved = True
                            break
                    if improved: break
                if improved: break
            if improved: break
            
    # Filter empty routes that might have been vacated
    routes = [r for r in routes if r]
    
    return ReplaceSolutionOperator(routes=routes), algorithm_data