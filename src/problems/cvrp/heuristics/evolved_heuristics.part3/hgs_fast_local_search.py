from src.problems.cvrp.components import ReplaceSolutionOperator
import numpy as np
import random
import time

def hgs_fast_local_search(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[ReplaceSolutionOperator, dict]:
    """
    SOTA-inspired Ultra-fast Local Search (First-Improvement, Route-level Cache).
    Aggregates Relocate, Swap, and 2-Opt in a single Python loop.
    Iterates nodes in randomized order against their granular neighborhoods.
    """
    dist = problem_state["distance_matrix"]
    neighbors = problem_state.get("nearest_neighbors")
    depot = problem_state["depot"]
    demands = problem_state["demands"]
    capacity = problem_state["capacity"]
    penalty_factor = problem_state.get("capacity_penalty_factor", 100.0)
    current_solution = problem_state["current_solution"]

    n_nodes = len(dist)
    if neighbors is None:
        neighbors = np.argsort(dist, axis=1)[:, 1:min(n_nodes, 41)]

    # Internal state tracking
    routes = [list(r) for r in current_solution.routes]
    node_to_route = {}
    node_to_pos = {}
    loads = [0] * len(routes)

    for vid, r in enumerate(routes):
        load = sum(demands[n] for n in r)
        loads[vid] = load
        for p, n in enumerate(r):
            node_to_route[n] = vid
            node_to_pos[n] = p

    def get_route_and_neighbors(vid, pos):
        r = routes[vid]
        prev_n = r[pos - 1] if pos > 0 else depot
        next_n = r[pos + 1] if pos + 1 < len(r) else depot
        return r, prev_n, next_n

    improved = True
    start_time = time.time()
    
    # Randomize the granular search iteration
    order = list(range(1, n_nodes))
    max_time = 5.0 # Max 5 seconds per intense LS
    
    while improved and time.time() - start_time < max_time:
        improved = False
        random.shuffle(order)
        
        for u in order:
            if u not in node_to_route: continue
            
            u_vid = node_to_route[u]
            u_pos = node_to_pos[u]
            u_route, prev_u, next_u = get_route_and_neighbors(u_vid, u_pos)
            u_demand = demands[u]

            for v in neighbors[u][:30]:
                if v == depot or v not in node_to_route: continue
                if u == v: continue
                
                v_vid = node_to_route[v]
                v_pos = node_to_pos[v]
                v_route, prev_v, next_v = get_route_and_neighbors(v_vid, v_pos)

                # --- 1. RELOCATE U after V ---
                # Check relocating U BEFORE V
                if u_vid != v_vid or (u_pos != v_pos - 1 and next_u != v):
                    d_dist = (
                        dist[prev_u][next_u] - dist[prev_u][u] - dist[u][next_u] +
                        dist[prev_v][u] + dist[u][v] - dist[prev_v][v]
                    )
                    d_pen = 0.0
                    if u_vid != v_vid:
                        o_p = max(0, loads[u_vid] - capacity) + max(0, loads[v_vid] - capacity)
                        n_p = max(0, loads[u_vid] - u_demand - capacity) + max(0, loads[v_vid] + u_demand - capacity)
                        d_pen = (n_p - o_p) * penalty_factor
                    
                    if d_dist + d_pen < -1e-4:
                        routes[u_vid].pop(u_pos)
                        if u_vid == v_vid and u_pos < v_pos:
                            v_pos -= 1
                        routes[v_vid].insert(v_pos, u)
                        if u_vid != v_vid:
                            loads[u_vid] -= u_demand
                            loads[v_vid] += u_demand
                        for p, n in enumerate(routes[u_vid]):
                            node_to_route[n] = u_vid
                            node_to_pos[n] = p
                        if u_vid != v_vid:
                            for p, n in enumerate(routes[v_vid]):
                                node_to_route[n] = v_vid
                                node_to_pos[n] = p
                        improved = True
                        break

                # --- 1b. RELOCATE U after V ---
                if u_vid != v_vid or (u_pos != v_pos + 1 and prev_u != v):
                    delta_dist = (
                        dist[prev_u][next_u] - dist[prev_u][u] - dist[u][next_u] +
                        dist[v][u] + dist[u][next_v] - dist[v][next_v]
                    )
                    if u_vid != v_vid:
                        old_pen = max(0, loads[u_vid] - capacity) + max(0, loads[v_vid] - capacity)
                        new_pen = max(0, loads[u_vid] - u_demand - capacity) + max(0, loads[v_vid] + u_demand - capacity)
                        delta_pen = (new_pen - old_pen) * penalty_factor
                    else:
                        delta_pen = 0.0
                        
                    if delta_dist + delta_pen < -1e-4:
                        # Apply Relocate
                        routes[u_vid].pop(u_pos)
                        if u_vid == v_vid and u_pos < v_pos:
                            v_pos -= 1
                        routes[v_vid].insert(v_pos + 1, u)
                        
                        if u_vid != v_vid:
                            loads[u_vid] -= u_demand
                            loads[v_vid] += u_demand
                            
                        # Rebuild cache for affected routes only
                        for p, n in enumerate(routes[u_vid]):
                            node_to_route[n] = u_vid
                            node_to_pos[n] = p
                        if u_vid != v_vid:
                            for p, n in enumerate(routes[v_vid]):
                                node_to_route[n] = v_vid
                                node_to_pos[n] = p
                                
                        improved = True
                        break # First improvement

                # --- 2. SWAP U and V ---
                v_demand = demands[v]
                if u_vid != v_vid or abs(u_pos - v_pos) > 1:
                    if u_vid == v_vid:
                        if u_pos > v_pos:
                            continue # symmetry
                        # Intra-route swap
                        if u_pos + 1 == v_pos:
                            delta_dist = dist[prev_u][v] + dist[v][u] + dist[u][next_v] - (dist[prev_u][u] + dist[u][v] + dist[v][next_v])
                        else:
                            delta_dist = (
                                dist[prev_u][v] + dist[v][next_u] - dist[prev_u][u] - dist[u][next_u] +
                                dist[prev_v][u] + dist[u][next_v] - dist[prev_v][v] - dist[v][next_v]
                            )
                        delta_pen = 0.0
                    else:
                        # Inter-route swap
                        delta_dist = (
                            dist[prev_u][v] + dist[v][next_u] - dist[prev_u][u] - dist[u][next_u] +
                            dist[prev_v][u] + dist[u][next_v] - dist[prev_v][v] - dist[v][next_v]
                        )
                        old_pen = max(0, loads[u_vid] - capacity) + max(0, loads[v_vid] - capacity)
                        new_pen = max(0, loads[u_vid] - u_demand + v_demand - capacity) + max(0, loads[v_vid] - v_demand + u_demand - capacity)
                        delta_pen = (new_pen - old_pen) * penalty_factor

                    if delta_dist + delta_pen < -1e-4:
                        # Apply Swap
                        routes[u_vid][u_pos] = v
                        routes[v_vid][v_pos] = u
                        node_to_route[u] = v_vid
                        node_to_pos[u] = v_pos
                        node_to_route[v] = u_vid
                        node_to_pos[v] = u_pos
                        if u_vid != v_vid:
                            loads[u_vid] = loads[u_vid] - u_demand + v_demand
                            loads[v_vid] = loads[v_vid] - v_demand + u_demand
                        improved = True
                        break # First improvement

                # --- 3. 2-OPT (Intra-route only for simplicity here, Inter-route is basically CROSS-exchange which is complex) ---
                if u_vid == v_vid and abs(u_pos - v_pos) > 1:
                    # To avoid symmetry, assume u_pos < v_pos
                    if u_pos > v_pos:
                        temp = u
                        u = v; v = temp
                        u_pos, v_pos = v_pos, u_pos
                        u_route, prev_u, next_u = get_route_and_neighbors(node_to_route[u], node_to_pos[u])
                        v_route, prev_v, next_v = get_route_and_neighbors(node_to_route[v], node_to_pos[v])

                    delta_dist = dist[u][v] + dist[next_u][next_v] - dist[u][next_u] - dist[v][next_v]
                    if delta_dist < -1e-4:
                        # Apply 2-opt (reverse segment next_u ... v)
                        routes[u_vid][u_pos+1:v_pos+1] = reversed(routes[u_vid][u_pos+1:v_pos+1])
                        for p in range(u_pos+1, v_pos+1):
                            node_to_pos[routes[u_vid][p]] = p
                        improved = True
                        break # First improvement
                        
            # --- 4. SWAP* (Advanced Inter-Route Swap) ---
                if not improved and u_vid != v_vid:
                    # Remove u from U, v from V
                    u_route_sans_u = u_route[:u_pos] + u_route[u_pos+1:]
                    v_route_sans_v = v_route[:v_pos] + v_route[v_pos+1:]
                    
                    if len(u_route_sans_u) == 0:
                        best_v_in_u_cost = 2.0 * dist[depot][v]
                        best_v_in_u_idx = 0
                    else:
                        prev_nodes = np.array([depot] + u_route_sans_u)
                        next_nodes = np.array(u_route_sans_u + [depot])
                        ins_costs = dist[prev_nodes, v] + dist[v, next_nodes] - dist[prev_nodes, next_nodes]
                        best_v_in_u_idx = int(np.argmin(ins_costs))
                        best_v_in_u_cost = float(ins_costs[best_v_in_u_idx])
                        
                    if len(v_route_sans_v) == 0:
                        best_u_in_v_cost = 2.0 * dist[depot][u]
                        best_u_in_v_idx = 0
                    else:
                        prev_nodes = np.array([depot] + v_route_sans_v)
                        next_nodes = np.array(v_route_sans_v + [depot])
                        ins_costs = dist[prev_nodes, u] + dist[u, next_nodes] - dist[prev_nodes, next_nodes]
                        best_u_in_v_idx = int(np.argmin(ins_costs))
                        best_u_in_v_cost = float(ins_costs[best_u_in_v_idx])
                        
                    removal_u_cost = dist[prev_u][next_u] - dist[prev_u][u] - dist[u][next_u]
                    removal_v_cost = dist[prev_v][next_v] - dist[prev_v][v] - dist[v][next_v]
                    
                    delta_dist = removal_u_cost + removal_v_cost + best_v_in_u_cost + best_u_in_v_cost
                    
                    old_pen = max(0, loads[u_vid] - capacity) + max(0, loads[v_vid] - capacity)
                    new_pen = max(0, loads[u_vid] - u_demand + v_demand - capacity) + max(0, loads[v_vid] - v_demand + u_demand - capacity)
                    delta_pen = (new_pen - old_pen) * penalty_factor
                    
                    if delta_dist + delta_pen < -1e-4:
                        # Apply SWAP*
                        # Safest way to apply is to pop/insert carefully and rebuild cache.
                        routes[u_vid].remove(u)
                        routes[v_vid].remove(v)
                        routes[u_vid].insert(best_v_in_u_idx, v)
                        routes[v_vid].insert(best_u_in_v_idx, u)
                        
                        loads[u_vid] = loads[u_vid] - u_demand + v_demand
                        loads[v_vid] = loads[v_vid] - v_demand + u_demand
                        
                        for p, n in enumerate(routes[u_vid]):
                            node_to_route[n] = u_vid
                            node_to_pos[n] = p
                        for p, n in enumerate(routes[v_vid]):
                            node_to_route[n] = v_vid
                            node_to_pos[n] = p
                        improved = True
                        break

                        
            if improved:

                        
                break # Restart granular...
                
    return ReplaceSolutionOperator(routes=routes), algorithm_data
