from src.problems.tsp.components import *
import numpy as np

def nearest_neighbor_b90a(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """ Enhanced nearest neighbor heuristic with dynamic starting point selection for the Traveling Salesman Problem (TSP).

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
            - "node_num" (int): The total number of nodes in the problem.
            - "std_dev_distance" (float): The standard deviation of distances to determine node selection strategy.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "unvisited_nodes" (list[int]): A list of integers representing the IDs of nodes that have not yet been visited.
            - "last_visited" (int or None): The last visited node, or None if no nodes have been visited yet.
        get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.
        kwargs: Hyper-parameters for the heuristic algorithm.
            - "threshold_factor" (float, default=0.70): Determines if a node's distance is significantly shorter.
            - "percentage_range" (float, default=0.20): Range within which distances are considered comparable.
            - "apply_2opt_frequency" (int, default=5): Frequency for applying the 2-opt heuristic.
            - "std_dev_threshold" (float, default=1000.0): Threshold for deciding starting node strategy based on distance variability.

    Returns:
        InsertOperator: The operator to insert the chosen node into the current solution.
        dict: Updated algorithm data if any.
    """
    # Extract necessary data
    distance_matrix = global_data["distance_matrix"]
    node_num = global_data["node_num"]
    std_dev_distance = global_data["std_dev_distance"]
    current_solution = state_data["current_solution"]
    unvisited_nodes = state_data["unvisited_nodes"]
    last_visited = state_data["last_visited"]

    # Set default hyper-parameters
    threshold_factor = kwargs.get("threshold_factor", 0.70)
    percentage_range = kwargs.get("percentage_range", 0.20)
    apply_2opt_frequency = kwargs.get("apply_2opt_frequency", 5)
    std_dev_threshold = kwargs.get("std_dev_threshold", 1000.0)

    # Initialize the tour if empty
    if not current_solution.tour:
        if not unvisited_nodes:
            return None, {}
        avg_distances = [np.mean([distance_matrix[i][j] for j in range(node_num) if i != j]) for i in range(node_num)]
        if std_dev_distance > std_dev_threshold:
            start_node = np.argmin(avg_distances)
        else:
            start_node = np.argsort(avg_distances)[1] if len(avg_distances) > 1 else np.argmin(avg_distances)
        return AppendOperator(start_node), {}

    # Return if no unvisited nodes remain
    if not unvisited_nodes:
        return None, {}

    # Periodically apply 2-opt heuristic
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

    # Calculate average distance from the last visited node to unvisited nodes
    avg_distance = np.mean([distance_matrix[last_visited][node] for node in unvisited_nodes])

    # Find nearest unvisited node to the last visited node
    nearest_node = min(unvisited_nodes, key=lambda node: distance_matrix[last_visited][node])
    nearest_distance = distance_matrix[last_visited][nearest_node]

    # Prioritize inserting the nearest node if its distance is significantly shorter
    if nearest_distance < threshold_factor * avg_distance:
        return InsertOperator(node=nearest_node, position=len(current_solution.tour)), {}

    # Evaluate nodes with comparable distances
    comparable_nodes = [node for node in unvisited_nodes if distance_matrix[last_visited][node] <= (1 + percentage_range) * nearest_distance]
    if len(comparable_nodes) <= 1:
        comparable_nodes = unvisited_nodes
    best_score = float('inf')
    best_node = None
    best_position = None

    for node in comparable_nodes:
        future_impact = sum([distance_matrix[node][unvisited] for unvisited in unvisited_nodes if unvisited != node])
        for i in range(len(current_solution.tour) + 1):
            if i == 0:
                next_node = current_solution.tour[0]
                immediate_impact = distance_matrix[node][next_node]
            elif i == len(current_solution.tour):
                prev_node = current_solution.tour[-1]
                immediate_impact = distance_matrix[prev_node][node]
            else:
                prev_node = current_solution.tour[i - 1]
                next_node = current_solution.tour[i]
                immediate_impact = distance_matrix[prev_node][node] + distance_matrix[node][next_node] - distance_matrix[prev_node][next_node]

            score = immediate_impact + state_data["visited_num"] / node_num / node_num * future_impact
            if score < best_score:
                best_score = score
                best_node = node
                best_position = i

    return InsertOperator(node=best_node, position=best_position), {}