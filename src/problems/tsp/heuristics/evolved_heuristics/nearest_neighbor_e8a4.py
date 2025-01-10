from src.problems.tsp.components import *
import numpy as np

def nearest_neighbor_e8a4(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """ Implements an enhanced nearest neighbor heuristic for the TSP problem.

    This heuristic starts from the node with the second lowest average distance to all other nodes when the tour is empty.
    It then proceeds by extending the tour using the nearest neighbor, nearest insertion, or cheapest insertion heuristics based on certain criteria.
    Periodically, the 2-opt heuristic is applied to untangle crossings and smooth the path.
    It includes additional rules for specific scenarios to enhance performance.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - distance_matrix (numpy.ndarray): A 2D array representing the distances between nodes.
            - node_num (int): The total number of nodes in the problem.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): An instance of the Solution class representing the current solution.
            - unvisited_nodes (list[int]): A list of integers representing the IDs of nodes that have not yet been visited.
            - last_visited (int or None): The last visited node, or None if no nodes have been visited yet.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution, and it will not modify the original solution.
        **kwargs: Hyper-parameters for the heuristic algorithm.
            - threshold_factor (float, default=0.70): The factor to determine whether a node's distance is significantly shorter than the average distance of unvisited nodes.
            - percentage_range (float, default=0.20): The percentage range within which multiple nodes' distances are considered comparable.
            - apply_2opt_frequency (int, default=5): The frequency (number of insertions) at which to apply the 2-opt heuristic.

    Returns:
        InsertOperator: The operator to insert the chosen node into the current solution.
        dict: Updated algorithm data if any.
    """
    distance_matrix = global_data["distance_matrix"]
    node_num = global_data["node_num"]
    current_solution = state_data["current_solution"]
    unvisited_nodes = state_data["unvisited_nodes"]
    last_visited = state_data["last_visited"]

    threshold_factor = kwargs.get("threshold_factor", 0.70)
    percentage_range = kwargs.get("percentage_range", 0.20)
    apply_2opt_frequency = kwargs.get("apply_2opt_frequency", 5)

    # If the current solution is empty, start from the node with the second lowest average distance to all other nodes
    if not current_solution.tour:
        avg_distances = [np.mean([distance_matrix[i][j] for j in range(node_num) if i != j]) for i in range(node_num)]
        sorted_nodes = np.argsort(avg_distances)
        start_node = sorted_nodes[1]  # Second lowest average distance
        nearest_node = min(unvisited_nodes, key=lambda node: distance_matrix[start_node][node])
        return AppendOperator(nearest_node), {}

    # If there are no unvisited nodes, return an empty operator
    if not unvisited_nodes:
        return None, {}

    # Apply the 2-opt heuristic periodically
    if len(current_solution.tour) % apply_2opt_frequency == 0 and len(current_solution.tour) > 2:
        best_delta = 0
        best_pair = None

        for i in range(len(current_solution.tour) - 1):
            for j in range(i + 2, len(current_solution.tour)):
                if j == len(current_solution.tour) - 1 and i == 0:
                    continue

                a, b = current_solution.tour[i], current_solution.tour[(i + 1) % len(current_solution.tour)]
                c, d = current_solution.tour[j], current_solution.tour[(j + 1) % len(current_solution.tour)]
                current_cost = distance_matrix[a][b] + distance_matrix[c][d]
                new_cost = distance_matrix[a][c] + distance_matrix[b][d]
                delta = new_cost - current_cost

                if delta < best_delta:
                    best_delta = delta
                    best_pair = (i + 1, j)

        if best_pair:
            return ReverseSegmentOperator([best_pair]), {}

    # Calculate the average distance of all unvisited nodes from the last visited node
    avg_distance = np.mean([distance_matrix[last_visited][node] for node in unvisited_nodes])

    # Find the nearest unvisited node to the last visited node
    nearest_node = min(unvisited_nodes, key=lambda node: distance_matrix[last_visited][node])
    nearest_distance = distance_matrix[last_visited][nearest_node]

    # Rule 1: If the distance to the nearest unvisited node is significantly shorter than the distance to other unvisited nodes, prioritize inserting the nearest node into the tour
    if nearest_distance < threshold_factor * avg_distance:
        return InsertOperator(node=nearest_node, position=len(current_solution.tour)), {}

    # Evaluate multiple unvisited nodes with comparable distances
    comparable_nodes = [node for node in unvisited_nodes if distance_matrix[last_visited][node] <= (1 + percentage_range) * nearest_distance]

    if len(current_solution.tour) == 3:
        best_connectivity = float('inf')
        best_node = None
        best_position = None

        for node in comparable_nodes:
            total_future_distance = 0
            for unvisited in unvisited_nodes:
                if unvisited != node:
                    total_future_distance += distance_matrix[node][unvisited]
            if total_future_distance < best_connectivity:
                best_connectivity = total_future_distance
                best_node = node
                best_position = len(current_solution.tour)

        if best_node is not None and best_position is not None:
            return InsertOperator(node=best_node, position=best_position), {}

    if len(comparable_nodes) > 1:
        best_increase = float('inf')
        best_node = None
        best_position = None

        for node in comparable_nodes:
            for i in range(len(current_solution.tour) + 1):
                if i == 0:
                    next_node = current_solution.tour[0]
                    cost_increase = distance_matrix[node][next_node]
                elif i == len(current_solution.tour):
                    prev_node = current_solution.tour[-1]
                    cost_increase = distance_matrix[prev_node][node]
                else:
                    prev_node = current_solution.tour[i - 1]
                    next_node = current_solution.tour[i]
                    cost_increase = distance_matrix[prev_node][node] + distance_matrix[node][next_node] - distance_matrix[prev_node][next_node]

                if cost_increase < best_increase:
                    best_increase = cost_increase
                    best_node = node
                    best_position = i

        if best_node is not None and best_position is not None:
            return InsertOperator(node=best_node, position=best_position), {}

    cheapest_cost_increase = float('inf')
    cheapest_node = None
    cheapest_position = None

    for node in unvisited_nodes:
        for i in range(len(current_solution.tour) + 1):
            if i == 0:
                next_node = current_solution.tour[0]
                cost_increase = distance_matrix[node][next_node]
            elif i == len(current_solution.tour):
                prev_node = current_solution.tour[-1]
                cost_increase = distance_matrix[prev_node][node]
            else:
                prev_node = current_solution.tour[i - 1]
                next_node = current_solution.tour[i]
                cost_increase = distance_matrix[prev_node][node] + distance_matrix[node][next_node] - distance_matrix[prev_node][next_node]

            if cost_increase < cheapest_cost_increase:
                cheapest_cost_increase = cost_increase
                cheapest_node = node
                cheapest_position = i

    if cheapest_node is not None and cheapest_position is not None:
        return InsertOperator(node=cheapest_node, position=cheapest_position), {}

    # Rule 2: If the tour is nearly complete and inserting the remaining unvisited nodes at the end of the tour results in a more compact and efficient path
    if len(unvisited_nodes) <= 3:
        return InsertOperator(node=nearest_node, position=len(current_solution.tour)), {}

    return InsertOperator(node=nearest_node, position=len(current_solution.tour)), {}