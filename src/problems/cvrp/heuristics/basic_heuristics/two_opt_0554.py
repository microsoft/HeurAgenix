from src.problems.cvrp.components import *

def two_opt_0554(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ReverseSegmentOperator, dict]:
    """
    Implements a 2-opt heuristic algorithm for the Capacitated Vehicle Routing Problem (CVRP).

    Args:
        global_data (dict): Contains the global data with necessary information like distance_matrix.
        state_data (dict): Contains the current state information with the current_solution.
        algorithm_data (dict): Contains the data necessary for this algorithm.

    Returns:
        TwoOptOperator: The operator that represents the best 2-opt move found.
        dict: Updated algorithm dictionary.
    """

    # Retrieve the necessary data from global_data
    distance_matrix = global_data["distance_matrix"]
    node_num = global_data["node_num"]
    depot = global_data["depot"]

    # Retrieve the current solution from state_data
    current_solution = state_data["current_solution"]
    visited_nodes = state_data["visited_nodes"]

    # Initialize variables for the best move found
    best_delta = 0
    best_move = None

    # Iterate over all routes to apply the 2-opt move
    for route_index, route in enumerate(current_solution.routes):
        for i in range(0, len(route)):
            for j in range(i + 1, len(route) + 1):
                # Calculate the cost difference for the current 2-opt move
                delta = two_opt_cost_change(distance_matrix, route, i, j, depot)
                # Check if this is the best move so far
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
    """Calculate the cost difference for a 2-opt move."""
    # Assuming the route is a circular tour
    A = depot if i == 0 else route[i - 1]
    B = depot if i == len(route) else route[i]
    C = depot if j == 0 else route[j - 1]
    D = depot if j == len(route) else route[j]
    d0 = distance_matrix[A][B] + distance_matrix[C][D]
    d1 = distance_matrix[A][C] + distance_matrix[B][D]

    # Return the cost difference
    return d1 - d0