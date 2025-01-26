from src.problems.tsp.components import *
import numpy as np

def farthest_insertion_54db(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """ Heuristic algorithm that selects the non-tour city that is farthest from any city in the current tour and inserts it where it causes the least increase in the tour cost.
    
    This heuristic also periodically applies the 2-opt heuristic to improve the solution.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - distance_matrix (numpy.ndarray): A 2D array representing the distances between nodes.
            - std_dev_distance (float): The standard deviation of distances.
            - node_num (int): The total number of nodes in the problem.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): The current solution of the TSP.
            - unvisited_nodes (list[int]): The list of nodes that have not been visited.
            - last_visited (int or None): The last visited node, or None if no nodes have been visited yet.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for new solution, and it will not modify the original solution.
        **kwargs: Hyper-parameters for the heuristic algorithm.
            - apply_2opt_frequency (int, default=5): The frequency (number of insertions) at which to apply the 2-opt heuristic.
            - high_disparity_threshold (float, default=1.0): The threshold for high disparity in distances.
            - low_disparity_threshold (float, default=0.5): The threshold for low disparity in distances.

    Returns:
        InsertOperator: The operator to insert the chosen node into the current solution.
        dict: Updated algorithm data if any.
    """
    distance_matrix = global_data["distance_matrix"]
    std_dev_distance = global_data["std_dev_distance"]
    node_num = global_data["node_num"]
    current_solution = state_data["current_solution"]
    unvisited_nodes = state_data["unvisited_nodes"]
    last_visited = state_data.get("last_visited")

    apply_2opt_frequency = kwargs.get("apply_2opt_frequency", 5)
    high_disparity_threshold = kwargs.get("high_disparity_threshold", 1.0)
    low_disparity_threshold = kwargs.get("low_disparity_threshold", 0.5)

    # If the current solution is empty, start from the node with the second lowest average distance to all other nodes
    if not current_solution.tour:
        avg_distances = [np.mean([distance_matrix[i][j] for j in range(node_num) if i != j]) for i in range(node_num)]
        sorted_nodes = np.argsort(avg_distances)
        start_node = sorted_nodes[1]  # Second lowest average distance
        nearest_node = min(unvisited_nodes, key=lambda node: distance_matrix[start_node][node])
        return AppendOperator(nearest_node), {}

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

    # Determine the application scope
    num_tour_nodes = len(current_solution.tour)

    # Initialize variables to track the farthest node and its insertion cost
    farthest_node = None
    max_distance_to_tour = -1
    min_insertion_cost = float('inf')
    insert_position = -1

    # Rule 1: When the number of visited nodes is greater than or equal to 60% of the total nodes and high disparity
    if num_tour_nodes >= 0.6 * node_num and std_dev_distance > high_disparity_threshold:
        # Use the least cost insertion heuristic
        for node in unvisited_nodes:
            for i in range(len(current_solution.tour)):
                next_i = (i + 1) % len(current_solution.tour)
                insertion_cost = (distance_matrix[current_solution.tour[i]][node] +
                                  distance_matrix[node][current_solution.tour[next_i]] -
                                  distance_matrix[current_solution.tour[i]][current_solution.tour[next_i]])
                if insertion_cost < min_insertion_cost:
                    min_insertion_cost = insertion_cost
                    farthest_node = node
                    insert_position = next_i

    # Rule 2: For fewer nodes in tour and high disparity
    elif num_tour_nodes < 0.6 * node_num and std_dev_distance > high_disparity_threshold:
        # Find the farthest node
        for node in unvisited_nodes:
            for tour_node in current_solution.tour:
                distance_to_tour_node = distance_matrix[node][tour_node]
                if distance_to_tour_node > max_distance_to_tour:
                    farthest_node = node
                    max_distance_to_tour = distance_to_tour_node
        
        # Find the position in the current tour where inserting the farthest node has the least cost
        for i in range(len(current_solution.tour)):
            next_i = (i + 1) % len(current_solution.tour)
            insertion_cost = (distance_matrix[current_solution.tour[i]][farthest_node] +
                              distance_matrix[farthest_node][current_solution.tour[next_i]] -
                              distance_matrix[current_solution.tour[i]][current_solution.tour[next_i]])
            if insertion_cost < min_insertion_cost:
                min_insertion_cost = insertion_cost
                insert_position = next_i

    # Rule 3: For fewer nodes in tour and low disparity
    elif num_tour_nodes < 0.6 * node_num and std_dev_distance <= low_disparity_threshold:
        # Use the nearest neighbor heuristic
        nearest_node = None
        min_distance_to_tour = float('inf')
        for node in unvisited_nodes:
            for tour_node in current_solution.tour:
                distance_to_tour_node = distance_matrix[node][tour_node]
                if distance_to_tour_node < min_distance_to_tour:
                    nearest_node = node
                    min_distance_to_tour = distance_to_tour_node
        
        # Find the position in the current tour where inserting the nearest node has the least cost
        for i in range(len(current_solution.tour)):
            next_i = (i + 1) % len(current_solution.tour)
            insertion_cost = (distance_matrix[current_solution.tour[i]][nearest_node] +
                              distance_matrix[nearest_node][current_solution.tour[next_i]] -
                              distance_matrix[current_solution.tour[i]][current_solution.tour[next_i]])
            if insertion_cost < min_insertion_cost:
                min_insertion_cost = insertion_cost
                insert_position = next_i
        farthest_node = nearest_node

    # Rule 4: For more nodes in tour and low disparity
    else:
        # Find the node that balances the distribution of nodes in the tour
        balanced_node = None
        balanced_insert_position = -1
        balanced_cost = float('inf')
        for node in unvisited_nodes:
            for i in range(len(current_solution.tour)):
                next_i = (i + 1) % len(current_solution.tour)
                insertion_cost = (distance_matrix[current_solution.tour[i]][node] +
                                  distance_matrix[node][current_solution.tour[next_i]] -
                                  distance_matrix[current_solution.tour[i]][current_solution.tour[next_i]])
                if insertion_cost < balanced_cost:
                    balanced_cost = insertion_cost
                    balanced_insert_position = next_i
                    balanced_node = node
        farthest_node = balanced_node
        insert_position = balanced_insert_position

    # If no valid farthest_node was found, return None
    if farthest_node is None:
        return None, {}

    # Create the insert operator with the farthest node and the best insertion position
    operator = InsertOperator(node=farthest_node, position=insert_position)
    return operator, {}