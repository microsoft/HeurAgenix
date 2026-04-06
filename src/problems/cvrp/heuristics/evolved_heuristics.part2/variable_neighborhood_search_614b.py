from src.problems.cvrp.components import *

def variable_neighborhood_search_614b(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[SwapOperator, dict]:
    """ 
    Re-purposed to: Fast Inter-Route Swap (KNN-accelerated).
    Exchanges exactly one node from vehicle_A with one node from vehicle_B.
    Evaluates only node pairs that are close to each other using the nearest_neighbors matrix,
    drastically reducing complexity from O(N^2) to O(N * K).
    Ensures capacity constraints are fully respected.

    Args:
        problem_state (dict): The dictionary containing the problem state.
        algorithm_data (dict): Algorithm-specific data.

    Returns:
        SwapOperator: The operator that modifies the solution, or None.
        dict: Empty dictionary.
    """
    current_solution = problem_state.get('current_solution')
    vehicle_loads = problem_state.get('vehicle_loads')
    capacity = problem_state.get('capacity')
    distance_matrix = problem_state.get('distance_matrix')
    demands = problem_state.get('demands')
    nearest_neighbors = problem_state.get('nearest_neighbors')

    best_operator = None
    best_cost_saving = 0.0

    # Build an O(1) lookup map for nodes to their route and position
    node_to_route = {}
    for vid, route in enumerate(current_solution.routes):
        for pos, node in enumerate(route):
             node_to_route[node] = (vid, pos)

    routes = current_solution.routes

    # Iterate over all nodes in the solution
    for nodeA, (vid_A, pos_A) in node_to_route.items():
        # Get KNN for nodeA
        if nearest_neighbors is not None:
             candidates = nearest_neighbors[nodeA]
        else:
             candidates = list(node_to_route.keys())

        # Node A's neighbors in its own route
        n_A = len(routes[vid_A])
        prev_A = routes[vid_A][(pos_A - 1) % n_A]
        next_A = routes[vid_A][(pos_A + 1) % n_A]
        demand_A = demands[nodeA]

        for nodeC in candidates:
            if nodeC not in node_to_route:
                 continue
            
            vid_B, pos_C = node_to_route[nodeC]

            # Only consider inter-route swaps
            if vid_A == vid_B:
                 continue
            
            demand_C = demands[nodeC]
            
            # Check capacity constraints for both vehicles
            if vehicle_loads[vid_A] - demand_A + demand_C > capacity:
                 continue
            if vehicle_loads[vid_B] - demand_C + demand_A > capacity:
                 continue

            n_B = len(routes[vid_B])
            prev_C = routes[vid_B][(pos_C - 1) % n_B]
            next_C = routes[vid_B][(pos_C + 1) % n_B]

            # Calculate cost change (Delta)
            # Remove A from vid_A, C from vid_B
            cost_increase = -distance_matrix[prev_A][nodeA] - distance_matrix[nodeA][next_A] \
                            -distance_matrix[prev_C][nodeC] - distance_matrix[nodeC][next_C]
            
            # Insert C into vid_A (at pos_A), A into vid_B (at pos_C)
            cost_increase += distance_matrix[prev_A][nodeC] + distance_matrix[nodeC][next_A] \
                             + distance_matrix[prev_C][nodeA] + distance_matrix[nodeA][next_C]

            cost_reduction = -cost_increase

            # If the swap leads to a better cost reduction, store it
            if cost_reduction > best_cost_saving + 1e-4:
                best_cost_saving = cost_reduction
                best_operator = SwapOperator(vehicle_id1=vid_A, position1=pos_A,
                                            vehicle_id2=vid_B, position2=pos_C)

    return best_operator, {}