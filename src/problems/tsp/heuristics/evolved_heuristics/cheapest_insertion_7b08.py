from src.problems.tsp.components import *
import numpy as np

def cheapest_insertion_7b08(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """ An enhanced heuristic algorithm for cheapest insertion in the Traveling Salesman Problem (TSP).

    This heuristic dynamically adjusts the clustering threshold or number of clusters based on a preliminary analysis of the dataset's node density or variance. 
    This allows the heuristic to adapt more effectively to different dataset characteristics.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - distance_matrix (numpy.ndarray): 2D array representing the distances between nodes.
            - node_num (int): The total number of nodes in the problem.
            - std_dev_distance (float): Standard deviation of distances to detect high variance.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): An instance of the Solution class representing the current solution.
            - unvisited_nodes (list[int]): A list of integers representing the IDs of nodes that have not yet been visited.
            - last_visited (int or None): The last visited node, or None if no nodes have been visited yet.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution, and it will not modify the original solution.
        kwargs: Hyper-parameters for the heuristic algorithm.
            - base_threshold_factor (float, default=0.90): The base factor to determine whether a node's distance is significantly shorter than the average distance of unvisited nodes.
            - std_dev_scaling_factor (float, default=0.01): The scaling factor to adjust the threshold based on standard deviation.
            - density_factor (float, default=1.0): Factor to adjust clustering based on dataset density.
            - clustering_std_dev_limit (float, default=1000.0): The standard deviation limit above which clustering will not be applied.

    Returns:
        InsertOperator: The operator to insert the chosen node into the current solution.
        dict: Updated algorithm data if any.
    """
    distance_matrix = global_data["distance_matrix"]
    node_num = global_data["node_num"]
    std_dev_distance = global_data["std_dev_distance"]
    current_solution = state_data["current_solution"]
    unvisited_nodes = state_data["unvisited_nodes"]
    last_visited = state_data.get("last_visited", None)

    base_threshold_factor = kwargs.get("base_threshold_factor", 0.90)
    std_dev_scaling_factor = kwargs.get("std_dev_scaling_factor", 0.01)
    density_factor = kwargs.get("density_factor", 1.0)
    clustering_std_dev_limit = kwargs.get("clustering_std_dev_limit", 1000.0)
    apply_2opt_frequency = kwargs.get("apply_2opt_frequency", 5)

    # Adjust threshold factor based on standard deviation
    threshold_factor = base_threshold_factor - std_dev_scaling_factor * std_dev_distance

    # If the current solution is empty, start from the node with the lowest average distance to all other nodes.
    if not current_solution.tour:
        avg_distances = [np.mean([distance_matrix[i][j] for j in range(node_num) if i != j]) for i in range(node_num)]
        start_node = unvisited_nodes[np.argmin(avg_distances)]
        return AppendOperator(start_node), {}

    # If there are no unvisited nodes, return None
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

    # Determine whether to apply clustering based on standard deviation
    if std_dev_distance <= clustering_std_dev_limit:
        # Calculate density-based clustering threshold
        avg_distance = np.mean([distance_matrix[last_visited][node] for node in unvisited_nodes])
        clustering_threshold = avg_distance * density_factor

        # Clustering nodes based on proximity logic
        cluster_nodes = [node for node in unvisited_nodes if distance_matrix[last_visited][node] < clustering_threshold] if last_visited is not None else unvisited_nodes

        # Early stage prioritization
        if len(current_solution.tour) < node_num // 2:
            # Rule: Insert node with the lowest average distance to all unvisited nodes within the cluster
            avg_distances_to_unvisited = {node: np.mean([distance_matrix[node][other] for other in unvisited_nodes if other != node]) for node in cluster_nodes}
            if avg_distances_to_unvisited:
                best_node = min(avg_distances_to_unvisited, key=avg_distances_to_unvisited.get)
                return InsertOperator(node=best_node, position=0), {}

    # General insertion strategy
    nearest_node = min(unvisited_nodes, key=lambda node: distance_matrix[last_visited][node])
    nearest_distance = distance_matrix[last_visited][nearest_node]

    if nearest_distance < threshold_factor * avg_distance:
        return InsertOperator(node=nearest_node, position=len(current_solution.tour)), {}

    # Original cheapest insertion heuristic
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

    # Fallback
    return InsertOperator(node=nearest_node, position=len(current_solution.tour)), {}