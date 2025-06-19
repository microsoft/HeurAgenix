from src.problems.tsp.components import *

def nearest_insertion_c1f0(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Implements the nearest insertion heuristic for the TSP problem. This heuristic starts with a sub-tour and at each step, inserts the non-tour city that is closest to any city in the current tour. The city is inserted at the position that results in the least cost increase.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "unvisited_nodes" (list[int]): A list of integers representing the IDs of nodes that have not yet been visited.
            - "visited_nodes" (list[int]): A list of integers representing the IDs of nodes that have been visited.

    Returns:
        InsertOperator: The operator to insert the nearest non-tour city into the current tour.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data from the problem state dictionaries
    distance_matrix = problem_state["distance_matrix"]
    current_solution = problem_state["current_solution"]
    unvisited_nodes = problem_state["unvisited_nodes"]
    visited_nodes = problem_state["visited_nodes"]

    # Initialize variables to store the best insertion
    best_increase = float('inf')
    best_unvisited_node = None
    best_position = None

    # If the current solution is empty, start from first unvisited node.
    if not current_solution.tour:
        return AppendOperator(unvisited_nodes[0]), {}

    # If there are no unvisited nodes, return an empty operator
    if not unvisited_nodes:
        return None, {}

    # Iterate over each unvisited node to find the nearest insertion
    for unvisited_node in unvisited_nodes:
        for i in range(len(visited_nodes)):
            # Calculate the position to insert the unvisited node
            position_to_insert = i + 1
            # Calculate the cost increase if we insert the unvisited node at the current position
            if position_to_insert < len(current_solution.tour):
                prev_node = current_solution.tour[i]
                next_node = current_solution.tour[position_to_insert]
                cost_increase = (distance_matrix[prev_node][unvisited_node] +
                                 distance_matrix[unvisited_node][next_node] -
                                 distance_matrix[prev_node][next_node])
            else:
                # If we are inserting at the end, the next node is the start of the tour
                prev_node = current_solution.tour[i]
                next_node = current_solution.tour[0]
                cost_increase = (distance_matrix[prev_node][unvisited_node] +
                                 distance_matrix[unvisited_node][next_node] -
                                 distance_matrix[prev_node][next_node])

            # Check if this is the best insertion found so far
            if cost_increase < best_increase:
                best_increase = cost_increase
                best_unvisited_node = unvisited_node
                best_position = position_to_insert

    # If a valid insertion was found, create the operator to perform the insertion
    if best_unvisited_node is not None and best_position is not None:
        insert_operator = InsertOperator(best_unvisited_node, best_position)
        return insert_operator, {}
    else:
        # If no valid insertion was found, return an empty operator
        return None, {}