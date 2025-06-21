# This file is generated generate_evaluation_function.py.
from src.problems.tsp.components import Solution

import numpy as np

def get_instance_problem_state(instance_data: dict) -> dict:
    """Extract instance problem state from instance data.

    Args:
        instance_data (dict): The dictionary contains the instance data.

    Returns:
        dict: The dictionary contains the instance problem state with:
            - average_distance (float): The average distance between all pairs of nodes.
            - min_distance (float): The minimum non-zero distance between any two nodes.
            - max_distance (float): The maximum distance between any two nodes.
            - std_dev_distance (float): The standard deviation of the distances.
            - density (float): The density of the graph, calculated as the ratio of non-zero distances to possible edges.
            - centroid (int): The node index that has the smallest sum of distances to all other nodes.
    """
    distance_matrix = instance_data["distance_matrix"]
    node_num = len(distance_matrix)
    
    # Compute the average distance while ignoring the diagonal (self-loops)
    average_distance = np.sum(distance_matrix) / (node_num * (node_num - 1))

    # Calculate the minimum non-zero distance
    min_distance = np.min(distance_matrix[np.nonzero(distance_matrix)])

    # Calculate the maximum distance
    max_distance = np.max(distance_matrix)

    # Calculate the standard deviation of the distances
    std_dev_distance = np.std(distance_matrix)

    return {
        "average_distance": average_distance,
        "min_distance": min_distance,
        "max_distance": max_distance,
        "std_dev_distance": std_dev_distance,
        "node_num": node_num
    }

def get_solution_problem_state(instance_data: dict, solution: Solution, get_key_value: callable) -> dict:
    """Extract solution problem state from instance data and solution.

    Args:
        instance_data (dict): The dictionary contains the instance data.
        solution (Solution): The target solution instance.
        get_key_value (callable): The function to get current_cost.

    Returns:
        dict: The dictionary contains the solution problem state with:
            - visited_nodes (list[int]): The list of visited node IDs.
            - unvisited_nodes (list[int]): The list of unvisited node IDs.
            - last_visited (int): The ID of the last visited node.
            - current_path_length (int): The number of nodes in the current path.
            - remaining_nodes (int): The number of nodes that remain unvisited.
            - average_edge_cost (float): The average cost per edge in the current solution.
            - last_edge_cost (float): The cost of the last edge added to the current solution.
            - std_dev_edge_cost (float): The standard deviation of edge costs in the current solution.
            - min_edge_cost_remaining (float): The minimum edge cost to any unvisited node from the last visited node.
            - max_edge_cost_remaining (float): The maximum edge cost to any unvisited node from the last visited node.
    """
    distance_matrix = instance_data["distance_matrix"]
    node_num = instance_data["node_num"]

    # Retrieve the tour from the solution, representing the order of nodes visited
    tour = solution.tour

    # Identifying nodes that have not been visited yet
    unvisited_nodes = [node for node in range(node_num) if node not in tour]

    # Identifying the last node that was visited
    last_visited = None if not solution.tour else solution.tour[-1]

    # Calculate number of nodes visited and unvisited
    visited_num = len(tour)
    unvisited_num = len(unvisited_nodes)

    # Use provided function to get the current cost of the tour
    current_cost = get_key_value(solution)

    # Calculate average cost per visited edge
    average_edge_cost = current_cost / visited_num if visited_num > 0 else float('0')
    # Retrieve the cost of the most recent edge
    last_edge_cost = distance_matrix[last_visited, tour[0]] if tour else 0
    # Compute edge costs for standard deviation calculation
    edge_costs = [distance_matrix[i, j] for i, j in zip(tour[:-1], tour[1:])] if len(tour) > 1 else [0]
    std_dev_edge_cost = np.std(edge_costs) if edge_costs else 0
    
    # Calculate minimum and maximum edge cost to any unvisited node from the last visited node
    min_edge_cost_remaining = np.min([distance_matrix[last_visited, j] for j in unvisited_nodes]) if unvisited_nodes else float('0')
    max_edge_cost_remaining = np.max([distance_matrix[last_visited, j] for j in unvisited_nodes]) if unvisited_nodes else 0
    
    return {
        "visited_nodes": tour,
        "unvisited_nodes": unvisited_nodes,
        "last_visited": last_visited,
        "visited_num": visited_num,
        "unvisited_num": unvisited_num,
        "average_edge_cost": average_edge_cost,
        "last_edge_cost": last_edge_cost,
        "std_dev_edge_cost": std_dev_edge_cost,
        "min_edge_cost_remaining": min_edge_cost_remaining,
        "max_edge_cost_remaining": max_edge_cost_remaining
    }

def get_observation_problem_state(problem_state: dict) -> dict:
    """Extract core problem state as observation.

    Args:
        problem_state (dict): The dictionary contains the problem state.

    Returns:
        dict: The dictionary contains the core problem state.
    """
    return {
        "visited_num": problem_state["visited_num"],
        "average_edge_cost": problem_state["average_edge_cost"]
    }