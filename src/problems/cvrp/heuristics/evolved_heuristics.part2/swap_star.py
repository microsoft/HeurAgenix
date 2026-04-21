from src.problems.cvrp.components import Solution, SwapStarOperator
import numpy as np

def swap_star(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[SwapStarOperator, dict]:
    """
    Advanced SWAP* inter-route operator.
    Evaluates swapping two nodes (A from Route 1, B from Route 2) and placing them
    into their BEST respective positions in the new routes.
    Includes Dynamic Penalty evaluation if capacity is violated.
    """
    distance_matrix = problem_state["distance_matrix"]
    nearest_neighbors = problem_state.get("nearest_neighbors", None)
    depot = problem_state["depot"]
    demands = problem_state["demands"]
    capacity = problem_state["capacity"]
    current_solution = problem_state["current_solution"]

    best_delta = 0
    best_move = None
    
    routes = current_solution.routes
    loads = current_solution.loads
    
    # We define a penalty factor (starts small, grows with stagnation)
    # If the hyper heuristic supports setting this, we get it from env.
    penalty_factor = problem_state.get("penalty_factor", 50.0) 

    # Helper function to find best insertion cost of node N into route R
    def evaluate_best_insertion(route, node_to_insert, node_to_remove):
        # We simulate route without node_to_remove
        temp_r = [n for n in route if n != node_to_remove]
        n_temp = len(temp_r)
        
        if n_temp == 0:
            # Route is empty except depot, inserting node_to_insert
            return 2 * distance_matrix[depot][node_to_insert], 0
            
        best_add_cost = float('inf')
        best_pos = 0
        
        # Check all insertion points
        # Because depot is implied at start and end
        for k in range(1, n_temp + 1):
            prev_n = depot if k == 0 else temp_r[k - 1]
            next_n = depot if k == n_temp else temp_r[k]
            
            add_cost = -distance_matrix[prev_n][next_n] + distance_matrix[prev_n][node_to_insert] + distance_matrix[node_to_insert][next_n]
            if add_cost < best_add_cost:
                best_add_cost = add_cost
                best_pos = k
                
        return best_add_cost, best_pos

    # If we have KNN, only check pairs within nearest neighbors
    num_routes = len(routes)
    for r1 in range(num_routes):
        for r2 in range(r1 + 1, num_routes):
            route1 = routes[r1]
            route2 = routes[r2]
            
            if not route1 or not route2: continue
            
            # Simple pairwise evaluation (can be accelerated with KNN filtering)
            for idx1, node1 in enumerate(route1):
                if node1 == depot: continue
                # Filter by KNN if available to boost speed
                if nearest_neighbors is not None:
                    # Check if any part of route2 has a neighbor to node1
                    if not any(n2 in nearest_neighbors[node1][:20] for n2 in route2):
                        continue
                        
                for idx2, node2 in enumerate(route2):
                    if node2 == depot: continue
                    # 1. Removal Savings
                    prev1 = depot if idx1 == 0 else route1[idx1 - 1]
                    next1 = depot if idx1 == len(route1) - 1 else route1[idx1 + 1]
                    rem_cost_1 = distance_matrix[prev1][node1] + distance_matrix[node1][next1] - distance_matrix[prev1][next1]
                    
                    prev2 = depot if idx2 == 0 else route2[idx2 - 1]
                    next2 = depot if idx2 == len(route2) - 1 else route2[idx2 + 1]
                    rem_cost_2 = distance_matrix[prev2][node2] + distance_matrix[node2][next2] - distance_matrix[prev2][next2]
                    
                    # 2. Find best insertion for node2 into route1 (without node1)
                    ins_cost_2_in_1, pos_2_in_1 = evaluate_best_insertion(route1, node2, node1)
                    
                    # 3. Find best insertion for node1 into route2 (without node2)
                    ins_cost_1_in_2, pos_1_in_2 = evaluate_best_insertion(route2, node1, node2)
                    
                    # 4. Total Distance Delta
                    dist_delta = (ins_cost_2_in_1 - rem_cost_1) + (ins_cost_1_in_2 - rem_cost_2)
                    
                    # 5. Capacity Penalty Delta evaluation
                    load1_new = loads[r1] - demands[node1] + demands[node2]
                    load2_new = loads[r2] - demands[node2] + demands[node1]
                    
                    pen1_old = max(0, loads[r1] - capacity)
                    pen2_old = max(0, loads[r2] - capacity)
                    pen1_new = max(0, load1_new - capacity)
                    pen2_new = max(0, load2_new - capacity)
                    
                    pen_delta = penalty_factor * ((pen1_new + pen2_new) - (pen1_old + pen2_old))
                    
                    total_delta = dist_delta + pen_delta
                    
                    if total_delta < best_delta - 1e-4:
                        best_delta = total_delta
                        best_move = (r1, node1, pos_1_in_2, r2, node2, pos_2_in_1)

    if best_move:
        r1, n1, p1_in_2, r2, n2, p2_in_1 = best_move
        # Note: Proper environment parsing required for SwapStarOperator delta application
        op = SwapStarOperator(
            vehicle_id1=r1, node1=n1, best_pos_for_1_in_2=p1_in_2,
            vehicle_id2=r2, node2=n2, best_pos_for_2_in_1=p2_in_1
        )
        return op, algorithm_data

    return None, algorithm_data
