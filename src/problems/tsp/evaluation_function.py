# This file is generated generate_evaluation_function.py and to renew the function, run "python generate_evaluation_function.py"

import numpy as np

def get_global_data_feature(global_data: dict) -> dict:
    """Extract features from the global data of a TSP instance.

    Args:
        global_data (dict): Contains the global static information data with:
            - "node_num" (int): The total number of nodes in the problem.
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.

    Returns:
        dict: A dictionary with the following keys and values:
            - "average_distance" (float): The average distance between all pairs of nodes.
            - "min_distance" (float): The minimum non-zero distance between any two nodes.
            - "max_distance" (float): The maximum distance between any two nodes.
            - "std_dev_distance" (float): The standard deviation of the distances.
            - "density" (float): The density of the graph, calculated as the ratio of non-zero distances to possible edges.
            - "centroid" (int): The node index that has the smallest sum of distances to all other nodes.
    """
    distance_matrix = global_data["distance_matrix"]
    node_num = global_data["node_num"]
    
    # Compute the average distance while ignoring the diagonal (self-loops)
    average_distance = np.sum(distance_matrix) / (node_num * (node_num - 1))

    # Calculate the minimum non-zero distance
    min_distance = np.min(distance_matrix[np.nonzero(distance_matrix)])

    # Calculate the maximum distance
    max_distance = np.max(distance_matrix)

    # Calculate the standard deviation of the distances
    std_dev_distance = np.std(distance_matrix)

    # Calculate the density of the graph
    density = np.count_nonzero(distance_matrix) / (node_num * (node_num - 1))

    # Find the centroid node
    centroid = np.argmin(np.sum(distance_matrix, axis=0))

    return {
        "average_distance": average_distance,
        "min_distance": min_distance,
        "max_distance": max_distance,
        "std_dev_distance": std_dev_distance,
        "density": density,
        "centroid": centroid
    }

import numpy as np

def get_state_data_feature(global_data: dict, state_data: dict) -> dict:
    """Extract features from the state data of a TSP instance.

    Args:
        global_data (dict): The global data dict containing:
            - "node_num" (int): The total number of nodes in the problem.
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
        state_data (dict): The state data dict containing:
            - "current_solution" (Solution): The current solution instance.
            - "visited_nodes" (list[int]): The list of visited node IDs.
            - "unvisited_nodes" (list[int]): The list of unvisited node IDs.
            - "current_cost" (int): The total cost of the current solution.
            - "last_visited" (int): The ID of the last visited node.
            - "validation_solution" (callable): Function to validate the solution.

    Returns:
        dict: A dictionary with the following keys and values representing the state features:
            - "current_path_length" (int): The number of nodes in the current path.
            - "remaining_nodes" (int): The number of nodes that remain unvisited.
            - "current_cost" (int): The total cost of the current solution.
            - "average_edge_cost" (float): The average cost per edge in the current solution.
            - "last_edge_cost" (float): The cost of the last edge added to the current solution.
            - "std_dev_edge_cost" (float): The standard deviation of edge costs in the current solution.
            - "solution_validity" (int): 1 if the current solution is valid, 0 otherwise.
            - "min_edge_cost_remaining" (float): The minimum edge cost to any unvisited node from the last visited node.
            - "max_edge_cost_remaining" (float): The maximum edge cost to any unvisited node from the last visited node.
    """
    distance_matrix = global_data["distance_matrix"]
    tour = state_data["current_solution"].tour
    current_cost = state_data["current_cost"]
    last_visited = state_data["last_visited"]
    unvisited_nodes = state_data["unvisited_nodes"]
    validation_solution = state_data["validation_solution"]
    
    current_path_length = len(tour)
    remaining_nodes = len(unvisited_nodes)
    
    average_edge_cost = current_cost / current_path_length if current_path_length > 0 else float('inf')
    last_edge_cost = distance_matrix[last_visited, tour[0]] if tour else 0
    edge_costs = [distance_matrix[i, j] for i, j in zip(tour[:-1], tour[1:])] if len(tour) > 1 else [0]
    std_dev_edge_cost = np.std(edge_costs) if edge_costs else 0
    solution_validity = int(validation_solution(state_data["current_solution"]))
    
    min_edge_cost_remaining = np.min([distance_matrix[last_visited, j] for j in unvisited_nodes]) if unvisited_nodes else float('inf')
    max_edge_cost_remaining = np.max([distance_matrix[last_visited, j] for j in unvisited_nodes]) if unvisited_nodes else 0
    
    return {
        "current_path_length": current_path_length,
        "remaining_nodes": remaining_nodes,
        "current_cost": current_cost,
        "average_edge_cost": average_edge_cost,
        "last_edge_cost": last_edge_cost,
        "std_dev_edge_cost": std_dev_edge_cost,
        "solution_validity": solution_validity,
        "min_edge_cost_remaining": min_edge_cost_remaining,
        "max_edge_cost_remaining": max_edge_cost_remaining
    }