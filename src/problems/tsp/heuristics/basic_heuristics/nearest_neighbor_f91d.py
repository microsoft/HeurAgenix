from src.problems.tsp.components import *

def nearest_neighbor_f91d(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Implements the nearest neighbor heuristic for the TSP problem. Starting from an arbitrary city, at each step extend the tour by moving from the current city to its nearest unvisited neighbor until all cities are visited.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "unvisited_nodes" (list[int]): A list of integers representing the IDs of nodes that have not yet been visited.
            - "last_visited" (int): The last visited node.

    Returns:
        InsertOperator: The operator to insert the nearest unvisited node into the current solution.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Retrieve necessary data from problem_state
    distance_matrix = problem_state["distance_matrix"]
    current_solution = problem_state["current_solution"]
    unvisited_nodes = problem_state["unvisited_nodes"]
    last_visited = problem_state["last_visited"]

    # If the current solution is empty, start from first unvisited node.
    if not current_solution.tour:
        start_node = unvisited_nodes[0]
        return AppendOperator(start_node), {}

    # If there are no unvisited nodes, return an empty operator
    if not unvisited_nodes:
        return None, {}

    # Find the nearest unvisited node to the last visited node
    nearest_node = None
    min_distance = float('inf')
    for node in unvisited_nodes:
        distance = distance_matrix[last_visited][node]
        if distance < min_distance:
            nearest_node = node
            min_distance = distance

    # If a nearest node is found, insert that node
    if nearest_node is not None:
        # Assuming we insert the nearest node at the end of the current solution
        position = len(current_solution.tour)
        return InsertOperator(node=nearest_node, position=position), {}
    else:
        # If no nearest node is found, return an empty operator
        return None, {}