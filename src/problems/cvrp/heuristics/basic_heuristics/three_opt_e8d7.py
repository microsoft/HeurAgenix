from src.problems.cvrp.components import *

def three_opt_e8d7(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ReverseSegmentOperator, dict]:
    """Implements a 3-opt heuristic algorithm for the Capacitated Vehicle Routing Problem (CVRP).

    Args:
        global_data (dict): Contains the global data with necessary information like distance_matrix.
        state_data (dict): Contains the current state information with the current_solution.
        algorithm_data (dict): Contains the data necessary for this algorithm.

    Returns:
        TwoOptOperator: The operator that represents the best 2-opt move found.
        dict: Updated algorithm dictionary.
    """
    # Retrieve data from global_data and state_data
    distance_matrix = global_data["distance_matrix"]
    depot = global_data["depot"]
    current_solution = state_data["current_solution"]

    # Initialize variables for the best move found
    best_delta = 0
    best_move = None

    # Iterate over all routes to apply the 3-opt move
    for route_index, route in enumerate(current_solution.routes):
        n = len(route)
        if n <= 3:
            continue  # Skip routes that are too short to apply 3-opt

        # Iterate through all combinations of three edges
        for i in range(n):
            for j in range(i + 2, n):
                for k in range(j + 2, n + (1 if i > 0 else 0)):
                    # Calculate the cost difference for the 3-opt move
                    moves, delta = calculate_3opt_moves(distance_matrix, route, i, j, k, depot)

                    # Check if this move is better than the best found so far
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (route_index, moves)
                        return ReverseSegmentOperator(route_index, moves), algorithm_data

    # If a beneficial move is found, create and return the corresponding operator
    if best_move:
        route_index, segments = best_move
        return ReverseSegmentOperator(route_index, segments), algorithm_data

    # If no beneficial move is found, return None
    return None, algorithm_data

def calculate_3opt_moves(distance_matrix, route, i, j, k, depot):
    """
    Calculate the cost difference for all 3-opt reconnection moves for the given segments and return the best move.
    """
    A = depot if i == 0 else route[i - 1]
    B = depot if i == len(route) else route[i]
    C = depot if j == 0 else route[j - 1]
    D = depot if j == len(route) else route[j]
    E = depot if k == 0 else route[k - 1]
    F = depot if k == len(route) else route[k]

    # Calculate the cost of the original segments
    original_cost = distance_matrix[A][B] + distance_matrix[C][D] + distance_matrix[E][F]

    # Initialize the list of all possible new segments and their costs
    possible_moves = [
        (distance_matrix[A][D] + distance_matrix[C][F] + distance_matrix[E][B], [(i, j), (j, k)]),  # Case 1: Two segments reversed
        (distance_matrix[A][E] + distance_matrix[D][B] + distance_matrix[C][F], [(i, j - 1), (i, k - 1)]),
        (distance_matrix[A][D] + distance_matrix[E][B] + distance_matrix[C][F], [(i, j), (j - 1, k - 1)]),
    ]


    # Find the move with the maximum cost reduction
    best_delta = 0
    best_move = None
    for cost, move in possible_moves:
        delta = cost - original_cost
        if delta < best_delta:
            best_delta = delta
            best_move = move

    # Return the best move and the corresponding cost reduction
    return best_move, best_delta