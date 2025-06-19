from src.problems.tsp.components import *

def greedy_algorithm_3ca7(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Greedy algorithm heuristic for TSP. Constructs a tour by repeatedly selecting the shortest edge and adding it to the tour.
    This implementation assumes that the Solution class represents a tour and that the InsertOperator is used to add new nodes to the Solution.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "unvisited_nodes" (list[int]): A list of integers representing the IDs of nodes that have not yet been visited.

    Returns:
        InsertOperator: The operator to insert the next node into the current solution.
        dict: Empty dictionary as this algorithm does not update the algorithm data.
    """
    distance_matrix = problem_state['distance_matrix']
    current_solution = problem_state['current_solution']
    unvisited_nodes = problem_state['unvisited_nodes']
    last_visited = problem_state["last_visited"]

    # If the current solution is empty, start from first unvisited node.
    if not current_solution.tour:
        return AppendOperator(unvisited_nodes[0]), {}

    # If solution is complete, return None.
    if len(unvisited_nodes) == 0:
        return None, {}

    # Find the shortest edge from the last node in the current solution to an unvisited node
    last_node = current_solution.tour[-1]
    min_distance = float('inf')
    next_node = None
    for node in unvisited_nodes:
        if distance_matrix[last_node][node] < min_distance:
            min_distance = distance_matrix[last_node][node]
            next_node = node

    # If no next node is found, return an empty operator
    if next_node is None:
        return None, {}

    # Find the best position to insert it
    best_position = 0
    best_increase = float('inf')
    for i in range(len(current_solution.tour) + 1):
        # Calculate the increase in cost if we insert the next node at position i
        if i == 0:
            prev_node = current_solution.tour[-1]
        else:
            prev_node = current_solution.tour[i - 1]
        if i == len(current_solution.tour):
            next_tour_node = current_solution.tour[0]
        else:
            next_tour_node = current_solution.tour[i]
        increase = (distance_matrix[prev_node][next_node] +
                    distance_matrix[next_node][next_tour_node] -
                    distance_matrix[prev_node][next_tour_node])
        if increase < best_increase:
            best_increase = increase
            best_position = i

    # Return the operator to insert the next node at the best position
    return InsertOperator(node=next_node, position=best_position), {}