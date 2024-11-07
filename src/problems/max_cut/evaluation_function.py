# This file is generated generate_evaluation_function.py and to renew the function, run "python generate_evaluation_function.py"

import numpy as np

def get_global_data_feature(global_data: dict) -> dict:
    """Extract features from global data for MaxCut problem.

    Args:
        global_data (dict): A dictionary containing:
            - "node_num" (int): Number of vertices in the graph.
            - "weight_matrix" (numpy.ndarray): Weights between nodes.

    Returns:
        dict: A dictionary with calculated features:
            - average_node_degree (float): The average degree of nodes in the graph.
            - edge_density (float): The density of edges in the graph.
            - average_edge_weight (float): The average weight of the edges.
            - max_edge_weight (float): The maximum edge weight.
            - min_edge_weight (float): The minimum non-zero edge weight.
            - standard_deviation_edge_weight (float): The standard deviation of the edge weights.
            - weighted_degree_distribution (numpy.ndarray): The weighted degree of each node.
    """
    node_num = global_data['node_num']
    weight_matrix = global_data['weight_matrix']

    # Ensure the weight matrix is a numpy array
    weight_matrix = np.array(weight_matrix)
    
    # Calculate the number of edges and the sum of edge weights
    nonzero_edges = np.count_nonzero(weight_matrix)
    total_edge_weight = np.sum(weight_matrix)

    # Calculate features
    average_node_degree = np.mean(np.count_nonzero(weight_matrix, axis=0))
    edge_density = nonzero_edges / (node_num * (node_num - 1))
    average_edge_weight = total_edge_weight / nonzero_edges
    max_edge_weight = np.max(weight_matrix)
    min_edge_weight = np.min(weight_matrix[weight_matrix.nonzero()])
    standard_deviation_edge_weight = np.std(weight_matrix[weight_matrix.nonzero()])
    weighted_degree_distribution = np.sum(weight_matrix, axis=0)

    # Construct the feature dictionary
    features = {
        'average_node_degree': average_node_degree,
        'edge_density': edge_density,
        'average_edge_weight': average_edge_weight,
        'max_edge_weight': max_edge_weight,
        'min_edge_weight': min_edge_weight,
        'standard_deviation_edge_weight': standard_deviation_edge_weight,
        'weighted_degree_distribution': weighted_degree_distribution
    }

    return features

import numpy as np

def get_state_data_feature(global_data: dict, state_data: dict) -> dict:
    """Extract features from state data for the MaxCut problem.

    Args:
        global_data (dict): Contains the global static information such as:
            - "node_num" (int): Total number of vertices in the graph.
            - "weight_matrix" (numpy.ndarray): Weight matrix between nodes.
        state_data (dict): Contains the current dynamic state data such as:
            - "current_solution" (Solution): Current solution instance.
            - "set_a_count" (int): Number of nodes in set A.
            - "set_b_count" (int): Number of nodes in set B.
            - "selected_nodes" (set[int]): Selected nodes.
            - "unselected_nodes" (set[int]): Unselected nodes.
            - "current_cut_value" (int or float): Weight of edges between sets A and B.

    Returns:
        dict: A dictionary with calculated features for the current state:
            - imbalance_ratio (float): Measure of balance between set A and set B.
            - cut_value (float): The total weight of edges between sets A and B.
            - average_cut_edge_weight (float): Average weight contribution of selected edges.
            - selected_nodes_ratio (float): Ratio of nodes that have been selected.
            - unselected_nodes_ratio (float): Ratio of nodes that have not been selected.
            - edge_weight_variance_within_sets (float): Variance of edge weights within each set.
            - boundary_node_ratio (float): Ratio of nodes that are adjacent to unselected nodes.
    """
    node_num = global_data['node_num']
    weight_matrix = global_data['weight_matrix']
    current_solution = state_data['current_solution']
    set_a_count = state_data['set_a_count']
    set_b_count = state_data['set_b_count']
    selected_nodes = state_data['selected_nodes']
    unselected_nodes = state_data['unselected_nodes']
    current_cut_value = state_data['current_cut_value']

    # Calculate features
    imbalance_ratio = abs(set_a_count - set_b_count) / node_num
    cut_value = current_cut_value
    average_cut_edge_weight = current_cut_value / len(selected_nodes) if selected_nodes else 0
    selected_nodes_ratio = len(selected_nodes) / node_num
    unselected_nodes_ratio = len(unselected_nodes) / node_num
    internal_edges = [weight_matrix[i][j] for i in current_solution.set_a for j in current_solution.set_a if i != j] + \
                     [weight_matrix[i][j] for i in current_solution.set_b for j in current_solution.set_b if i != j]
    edge_weight_variance_within_sets = np.var(internal_edges) if internal_edges else 0
    
    # Calculate boundary nodes (nodes in selected_nodes that have an edge to unselected_nodes)
    boundary_nodes = len([node for node in selected_nodes if any(neighbor in unselected_nodes for neighbor in np.nonzero(weight_matrix[node])[0])])
    boundary_node_ratio = boundary_nodes / node_num

    # Construct the feature dictionary
    features = {
        'imbalance_ratio': imbalance_ratio,
        'cut_value': cut_value,
        'average_cut_edge_weight': average_cut_edge_weight,
        'selected_nodes_ratio': selected_nodes_ratio,
        'unselected_nodes_ratio': unselected_nodes_ratio,
        'edge_weight_variance_within_sets': edge_weight_variance_within_sets,
        'boundary_node_ratio': boundary_node_ratio
    }

    return features