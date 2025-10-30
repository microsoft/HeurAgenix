from src.problems.tsp.components import *
import random

def random_pairwise_insertion_7493(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Randomized two-node pairwise insertion. Samples two distinct unvisited nodes uniformly and evaluates their simultaneous placement by scanning all position pairs (i, j) in 0..L subject to i ≠ j and |i − j| ≠ 1 to avoid adjacent placements and immediate mutual interference. The joint objective is the sum of independent single-node marginal costs computed on the same base tour (no re-evaluation after the first insertion, no interaction terms, no tie policy). Marginal cost model:
    - For position p > 0: replace edge (tour[p−1], tour[p%L]) with (tour[p−1], node) + (node, tour[p%L]) using circular wrap-around.
    - For position p = 0: add only d[node, tour[0]] (head-biased, open-path-like treatment).
    This mix yields a circular model except at the head, and is compatible with asymmetric distances. Assumes a non-empty current tour and at least two unvisited nodes. Time complexity per sampled pair: O((L+1)^2); negligible extra memory; no algorithm_data dependence.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "unvisited_nodes" (list[int]): A list of integers representing the IDs of nodes that have not yet been visited.

    Returns:
        InsertOperator: The operator to insert two nodes into the solution.
        dict: Empty dictionary as no algorithm data is updated.
    """

    # Extract necessary data from problem_state
    distance_matrix = problem_state["distance_matrix"]
    current_solution = problem_state["current_solution"]
    unvisited_nodes = problem_state["unvisited_nodes"]

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