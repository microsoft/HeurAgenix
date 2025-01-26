from src.problems.tsp.components import *
import random

def random_successive_insertion_57b4(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Constructive heuristic that builds a tour by successively inserting random unvisited nodes into the current tour at a position that minimizes the increase in tour length.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "unvisited_nodes" (list[int]): A list of integers representing the IDs of nodes that have not yet been visited.
            - "visited_nodes" (list[int]): A list of integers representing the IDs of nodes that have been visited.

    Returns:
        InsertOperator: The operator to insert a randomly chosen unvisited node into the current solution.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data from the global and state dictionaries
    distance_matrix = global_data["distance_matrix"]
    current_solution = state_data["current_solution"]
    unvisited_nodes = state_data["unvisited_nodes"]
    visited_nodes = state_data["visited_nodes"]

    # If there are no unvisited nodes, return an empty operator
    if not unvisited_nodes:
        return None, {}

    # Randomly select an unvisited node
    random_node = random.choice(unvisited_nodes)

    # If the current solution is empty or has only one node, append the action
    if not current_solution.tour or len(current_solution.tour) == 1:
        return InsertOperator(random_node, position=0), {}

    # Find the best position to insert the node that minimizes the increase in tour length
    best_increase = float('inf')
    best_position = 0
    for i in range(len(visited_nodes)):
        # Calculate the increase in tour length if inserted at the current position
        prev_node = visited_nodes[i]
        next_node = visited_nodes[(i + 1) % len(visited_nodes)]
        increase = (distance_matrix[prev_node][random_node] +
                    distance_matrix[random_node][next_node] -
                    distance_matrix[prev_node][next_node])
        if increase < best_increase:
            best_increase = increase
            best_position = i + 1

    # Create and return the insert operator with the chosen node and position
    return InsertOperator(random_node, position=best_position), {}