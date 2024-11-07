from src.problems.tsp.components import *

def _2opt_89aa(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ReverseSegmentOperator, dict]:
    """
    The 2-opt heuristic seeks to untangle crossings and smooth the path by swapping two non-adjacent edges and reconnecting the resulting segments. Through repeated application of these edge swaps, the 2-opt algorithm converges towards a more efficient route, often leading to a substantial improvement over the initial solution. It is a simple yet powerful method for local optimization in the context of the TSP.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "current_cost" (int): The total cost of current solution.

    Returns:
        ReverseSegmentOperator: The operator that reverse two nodes in the solution to achieve a shorter tour.
        dict: Empty dictionary as this algorithm does not update algorithm_data.
    """
    distance_matrix = global_data["distance_matrix"]
    current_solution = state_data["current_solution"]
    current_cost = state_data["current_cost"]

    # Best improvement setup
    best_delta = 0
    best_pair = None

    # Iterate over all pairs of indices to consider removing
    for i in range(len(current_solution.tour) - 1):  
            for j in range(i + 2, len(current_solution.tour)):  
                if j == len(current_solution.tour) - 1 and i == 0:  
                    continue

                # Calculate the cost difference if these two edges are removed and reconnected
                a, b = current_solution.tour[i], current_solution.tour[(i + 1) % len(current_solution.tour)]
                c, d = current_solution.tour[j], current_solution.tour[(j + 1) % len(current_solution.tour)]
                current_cost = distance_matrix[a][b] + distance_matrix[c][d]
                new_cost = distance_matrix[a][c] + distance_matrix[b][d]
                delta = new_cost - current_cost

                # Check for an improvement
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (i + 1, j)

    # If an improvement has been found, create and return the corresponding SwapOperator
    if best_pair:
        return ReverseSegmentOperator([best_pair]), {}
    else:
        # No improvement found, return an empty operator
        return None, {}