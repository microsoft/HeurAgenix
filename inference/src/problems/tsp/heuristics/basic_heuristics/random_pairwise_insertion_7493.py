from src.problems.tsp.components import *
import random

def random_pairwise_insertion_7493(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Random Pairwise Insertion heuristic for the Traveling Salesman Problem. This heuristic selects two unvisited nodes at random and inserts them into the current tour at positions that result in the least increase in tour length. It returns an InsertOperator with the selected nodes and their positions.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "unvisited_nodes" (list[int]): A list of integers representing the IDs of nodes that have not yet been visited.

    Returns:
        InsertOperator: The operator to insert two nodes into the solution.
        dict: Empty dictionary as no algorithm data is updated.
    """

    # Extract necessary data from global_data and state_data
    distance_matrix = global_data["distance_matrix"]
    current_solution = state_data["current_solution"]
    unvisited_nodes = state_data["unvisited_nodes"]

    # Check if there are at least two unvisited nodes to insert
    if len(unvisited_nodes) < 2:
        return None, {}

    # Randomly select two distinct unvisited nodes
    node_a, node_b = random.sample(unvisited_nodes, 2)

    # Initialize variables to track the best insertion cost and positions
    best_cost_increase = float('inf')
    best_positions = (None, None)

    # Consider current solution as a circular list, hence range(len + 1)
    for insert_position_a in range(len(current_solution.tour) + 1):
        for insert_position_b in range(len(current_solution.tour) + 1):
            # Avoid re-evaluating the same pair or consecutive positions
            if insert_position_a == insert_position_b or \
               abs(insert_position_a - insert_position_b) == 1:
                continue

            # Calculate the cost increase for inserting at the current positions
            cost_increase = calculate_insertion_cost(distance_matrix, current_solution.tour,
                                                     node_a, node_b, insert_position_a, insert_position_b)

            # Update the best cost and positions if this is a better insertion
            if cost_increase < best_cost_increase:
                best_cost_increase = cost_increase
                best_positions = (insert_position_a, insert_position_b)

    # Create the InsertOperator with the best positions found
    if best_positions[0] and best_positions[1]:
        if best_positions[0] < best_positions[1]:
            # Ensure that node_a is inserted first if it comes before node_b
            insert_operator = InsertOperator(node_a, best_positions[0])
            insert_operator = InsertOperator(node_b, best_positions[1])
        else:
            # Ensure that node_b is inserted first if it comes before node_a
            insert_operator = InsertOperator(node_b, best_positions[1])
            insert_operator = InsertOperator(node_a, best_positions[0])
        return insert_operator, {}
    else:
        return None, {}
    

def calculate_insertion_cost(distance_matrix, tour, node_a, node_b, pos_a, pos_b):
    """
    Calculate the cost increase for inserting two nodes at specified positions in the tour.

    Args:
        distance_matrix (numpy.ndarray): The distances between nodes.
        tour (list[int]): The current list of nodes in the solution.
        node_a (int): The first node to insert.
        node_b (int): The second node to insert.
        pos_a (int): The position to insert the first node.
        pos_b (int): The position to insert the second node.

    Returns:
        float: The cost increase for the insertion.
    """
    # Calculate the cost increase for inserting node_a
    if pos_a == 0:
        cost_increase_a = distance_matrix[node_a][tour[0]]
    else:
        cost_increase_a = distance_matrix[tour[pos_a - 1]][node_a] + \
                          distance_matrix[node_a][tour[pos_a % len(tour)]] - \
                          distance_matrix[tour[pos_a - 1]][tour[pos_a % len(tour)]]

    # Calculate the cost increase for inserting node_b
    if pos_b == 0:
        cost_increase_b = distance_matrix[node_b][tour[0]]
    else:
        cost_increase_b = distance_matrix[tour[pos_b - 1]][node_b] + \
                          distance_matrix[node_b][tour[pos_b % len(tour)]] - \
                          distance_matrix[tour[pos_b - 1]][tour[pos_b % len(tour)]]

    # Return the total cost increase
    return cost_increase_a + cost_increase_b