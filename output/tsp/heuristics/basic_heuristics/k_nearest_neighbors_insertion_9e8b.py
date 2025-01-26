from src.problems.tsp.components import *

def k_nearest_neighbors_insertion_9e8b(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, k: int = 1, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Heuristic that builds a tour by inserting the nearest unvisited neighbor to the last visited node.
    If multiple nearest neighbors are considered (k > 1), one is chosen based on a certain criterion.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "unvisited_nodes" (list[int]): A list of integers representing the IDs of nodes that have not yet been visited.
            - "last_visited" (int): The last visited node in the current solution.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. This algorithm does not use algorithm_data.
        k (int): The number of nearest neighbors to consider for insertion. Defaults to 1.

    Returns:
        InsertOperator: The operator to insert the chosen node into the current solution.
        dict: Empty dictionary as this algorithm does not update algorithm_data.
    """
    # Extract necessary data from global_data and state_data
    distance_matrix = global_data["distance_matrix"]
    current_solution = state_data["current_solution"]
    unvisited_nodes = state_data["unvisited_nodes"]
    last_visited = state_data["last_visited"]

    # If the current solution is empty, start from first unvisited node.
    if not current_solution.tour:
        return AppendOperator(unvisited_nodes[0]), {}

    # If there are no unvisited nodes, return an empty operator
    if not unvisited_nodes:
        return None, {}

    # Initialize variables to store the nearest neighbor and its insertion cost
    nearest_neighbor = None
    min_distance = float('inf')

    # Find the nearest unvisited neighbor or one of the k-nearest based on a criterion
    for unvisited in unvisited_nodes:
        distance = distance_matrix[last_visited][unvisited]
        if distance < min_distance:
            nearest_neighbor = unvisited
            min_distance = distance

    # If no unvisited neighbor is found, return an empty operator
    if nearest_neighbor is None:
        return None, {}


    # Determine the position to insert the node
    # For simplicity, we insert at the end of the current solution
    # This can be modified to insert at a specific position based on additional criteria
    insert_position = len(current_solution.tour)

    # Create and return the InsertOperator with the chosen node and position
    return InsertOperator(node=nearest_neighbor, position=insert_position), {}