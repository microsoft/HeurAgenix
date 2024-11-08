# This file is generated generate_evaluation_function.py and to renew the function, run "python generate_evaluation_function.py"

import numpy as np

def get_global_data_feature(global_data: dict) -> dict:
    """Extract features from global data for CVRP.
    
    Args:
        global_data (dict): The global data dict containing global static information.
    
    Returns:
        dict: A dictionary with calculated feature values.
    """
    # Calculate the average demand per node
    average_demand = np.mean(global_data["demands"])
    
    # Calculate the variance in demand across all nodes
    demand_variance = np.var(global_data["demands"])
    
    # Extract the upper triangle of the distance matrix to avoid duplicate distances
    upper_triangle_indices = np.triu_indices(global_data["node_num"], k=1)
    upper_triangle_distances = global_data["distance_matrix"][upper_triangle_indices]
    
    # Calculate the average distance between nodes
    average_distance = np.mean(upper_triangle_distances)
    
    # Find the maximum distance between any two nodes
    max_distance = np.max(upper_triangle_distances)
    
    # Find the minimum distance between any two nodes
    min_distance = np.min(upper_triangle_distances)
    
    # Calculate the variance of the distances
    distance_variance = np.var(upper_triangle_distances)
    
    # Compute vehicle capacity utilization
    vehicle_capacity_utilization = np.sum(global_data["demands"]) / (global_data["vehicle_num"] * global_data["capacity"])
    
    # Compute the ratio of the number of nodes to the number of vehicles
    node_to_vehicle_ratio = global_data["node_num"] / global_data["vehicle_num"]
    
    # Create a dictionary with all the features
    features = {
        "average_demand": average_demand,
        "demand_variance": demand_variance,
        "average_distance": average_distance,
        "max_distance": max_distance,
        "min_distance": min_distance,
        "distance_variance": distance_variance,
        "vehicle_capacity_utilization": vehicle_capacity_utilization,
        "node_to_vehicle_ratio": node_to_vehicle_ratio,
    }
    
    return features

import numpy as np

def get_state_data_feature(global_data: dict, state_data: dict) -> dict:
    """Extract features from state data for CVRP.
    
    Args:
        global_data (dict): The global data containing static information of the CVRP.
        state_data (dict): The state data containing dynamic information of the current CVRP solution.
    
    Returns:
        dict: A dictionary with calculated feature values representing the current state of the solution.
    """
    # Calculate route lengths and costs
    route_lengths = [len(route) for route in state_data["current_solution"].routes]
    route_costs = [sum(global_data["distance_matrix"][route[i]][route[i+1]] 
                       for i in range(len(route)-1)) 
                   for route in state_data["current_solution"].routes 
                   if len(route) > 1]
    
    # Calculate average route length
    average_route_length = np.mean(route_lengths) if route_lengths else 0
    
    # Calculate max and min route length
    max_route_length = max(route_lengths, default=0)
    min_route_length = min(route_lengths, default=0)
    
    # Calculate standard deviation of route length
    std_dev_route_length = np.std(route_lengths) if route_lengths else 0
    
    # Calculate average route cost
    average_route_cost = np.mean(route_costs) if route_costs else 0
    
    # Calculate total demand served
    total_demand_served = sum(global_data["demands"][node] for route in state_data["current_solution"].routes for node in route)
    
    # Calculate average vehicle load
    average_vehicle_load = np.mean(state_data["vehicle_loads"]) if state_data["vehicle_loads"] else 0
    
    # Calculate average remaining vehicle capacity
    average_remaining_vehicle_capacity = np.mean(state_data["vehicle_remaining_capacity"]) if state_data["vehicle_remaining_capacity"] else global_data["capacity"]
    
    # Count the number of unvisited nodes
    number_of_unvisited_nodes = len(state_data["unvisited_nodes"])
    
    # Calculate average unvisited node demand
    average_unvisited_node_demand = np.mean(global_data["demands"][state_data["unvisited_nodes"]]) if state_data["unvisited_nodes"] else 0
    
    # Calculate total remaining demand
    total_remaining_demand = sum(global_data["demands"][node] for node in state_data["unvisited_nodes"])
    
    # Check solution validity
    solution_validity = state_data["validation_solution"](state_data["current_solution"])
    
    # Create a dictionary with all the features
    features = {
        "average_route_length": average_route_length,
        "max_route_length": max_route_length,
        "min_route_length": min_route_length,
        "std_dev_route_length": std_dev_route_length,
        "average_route_cost": average_route_cost,
        "total_demand_served": total_demand_served,
        "average_vehicle_load": average_vehicle_load,
        "average_remaining_vehicle_capacity": average_remaining_vehicle_capacity,
        "number_of_unvisited_nodes": number_of_unvisited_nodes,
        "average_unvisited_node_demand": average_unvisited_node_demand,
        "total_remaining_demand": total_remaining_demand,
        "solution_validity": solution_validity,
    }
    
    return features