from src.problems.cvrp.components import *

def two_opt_0554(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[ReverseSegmentOperator, dict]:
    """
    Intra-route 2-opt accelerated with K-Nearest Neighbors (Granular Neighborhood).
    """

    # Retrieve the necessary data from problem_state
    distance_matrix = problem_state["distance_matrix"]
    nearest_neighbors = problem_state.get("nearest_neighbors", None)
    depot = problem_state["depot"]

    current_solution = problem_state["current_solution"]

    # Initialize variables for the best move found
    best_delta = 0
    best_move = None

    # Pre-build node to route & index mapping for O(1) lookups
    node_to_route_idx = {}
    for route_index, route in enumerate(current_solution.routes):
        for idx, node in enumerate(route):
            node_to_route_idx[node] = (route_index, idx)

    # Fast Granular Search if KNN is available
    if nearest_neighbors is not None:
        for route_index, route in enumerate(current_solution.routes):
            n = len(route)
            if n <= 2: continue
            
            for i in range(n):
                A = route[(i - 1) % n]
                B = route[i] # B is the start of the segment to reverse
                
                # Instead of iterating all j, we only check if A's nearest neighbors can be connected to C (which is j-1)
                # We want to form edge (A, C) replacing (A, B).
                # So we look for C in nearest_neighbors[A]
                for C in nearest_neighbors[A]:
                    if C not in node_to_route_idx: continue
                    r_idx, c_idx = node_to_route_idx[C]
                    if r_idx != route_index: continue # Intra-route only
                    
                    # j-1 = c_idx -> j = c_idx + 1
                    j = c_idx + 1
                    
                    # Ensure valid segment
                    if i < j:
                        delta = two_opt_cost_change(distance_matrix, route, i, j, depot)
                        if delta < best_delta - 1e-4:
                            best_delta = delta
                            best_move = (route_index, [(i, j - 1)])
                    elif j < i: # Wraparound case (be careful with indices here, stick to simple non-wrap for now to match original)
                        pass
                        
    else:
        # Fallback to O(N^2) if KNN is missing
        for route_index, route in enumerate(current_solution.routes):
            for i in range(1, len(route)):
                for j in range(i + 1, len(route) + 1):
                    delta = two_opt_cost_change(distance_matrix, route, i, j, depot)
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (route_index, [(i, j - 1)])

    # If a beneficial move is found, create and return the corresponding operator
    if best_move:
        route_index, move_pair = best_move
        return ReverseSegmentOperator(route_index, move_pair), algorithm_data

    # If no beneficial move is found, return None
    return None, algorithm_data

def two_opt_cost_change(distance_matrix, route, i, j, depot):
    """Calculate the cost difference for a 2-opt move on circular CVRP route [depot, c1, ..., cn]."""
    n = len(route)
    A = route[(i - 1) % n]
    B = route[i % n]
    C = route[(j - 1) % n]
    D = route[j % n]  # wraps to depot
    d0 = distance_matrix[A][B] + distance_matrix[C][D]
    d1 = distance_matrix[A][C] + distance_matrix[B][D]

    # Return the cost difference
    return d1 - d0