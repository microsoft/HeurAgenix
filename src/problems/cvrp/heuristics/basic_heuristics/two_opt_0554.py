from src.problems.cvrp.components import *

def two_opt_0554(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[ReverseSegmentOperator, dict]:
    """
    Implements a 2-opt heuristic algorithm for the Capacitated Vehicle Routing Problem (CVRP).

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
            - "depot" (int): The index for depot node.
            - "current_solution" (Solution): The current set of routes for all vehicles.

    Returns:
        TwoOptOperator: The operator that represents the best 2-opt move found.
        dict: Updated algorithm dictionary.
    """

    # Retrieve the necessary data from problem_state
    distance_matrix = problem_state["distance_matrix"]
    depot = problem_state["depot"]

    current_solution = problem_state["current_solution"]

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
    n = len(route)
    A = route[(i - 1) % n]
    B = route[i % n]
    C = route[(j - 1) % n]
    D = route[j % n]
    d0 = distance_matrix[A][B] + distance_matrix[C][D]
    d1 = distance_matrix[A][C] + distance_matrix[B][D]

    # Return the cost difference
    return d1 - d0