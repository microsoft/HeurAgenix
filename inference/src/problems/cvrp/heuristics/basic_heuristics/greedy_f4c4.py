from src.problems.cvrp.components import Solution, InsertOperator

def greedy_f4c4(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Greedy heuristic algorithm for the CVRP.
    This algorithm attempts to construct a solution by iteratively adding the closest unvisited node to a vehicle's route until all nodes are visited or the vehicle reaches its capacity.
    It starts with an empty route for each vehicle and selects nodes based on the shortest distance from the last node in the route.

    Args:
        global_data (dict): Contains the global problem data, including:
            - "distance_matrix" (numpy.ndarray): The distances between nodes.
            - "demands" (numpy.ndarray): The demand of each node.
            - "vehicle_num" (int): The total number of vehicles.
            - "capacity" (int): The capacity for each vehicle.
        
        state_data (dict): Contains the current state of the solution, including:
            - "unvisited_nodes" (list[int]): Nodes that have not yet been visited.
            - "vehicle_remaining_capacity" (list[int]): The remaining capacity for each vehicle.

    Returns:
        InsertOperator: The operator that adds a node to a vehicle's route.
        dict: Empty dictionary as the algorithm does not update any algorithm-specific data.
    """
    
    distance_matrix = global_data['distance_matrix']
    demands = global_data['demands']
    vehicle_num = global_data['vehicle_num']
    vehicle_capacity = global_data['capacity']
    
    unvisited_nodes = state_data['unvisited_nodes']
    vehicle_remaining_capacity = state_data['vehicle_remaining_capacity']
    
    # Start with the first unvisited node and the first vehicle
    for vehicle_id in range(vehicle_num):
        if not unvisited_nodes or vehicle_remaining_capacity[vehicle_id] <= 0:
            continue  # Skip to the next vehicle if there are no nodes left or no capacity.
        
        # Initialize a temporary variable to find the closest unvisited node
        closest_node = None
        closest_distance = float('inf')
        
        last_visited_node = 0  # Assuming the depot ID is 0
        for node in unvisited_nodes:
            if demands[node] <= vehicle_remaining_capacity[vehicle_id]:
                distance = distance_matrix[last_visited_node][node]
                if distance < closest_distance:
                    closest_distance = distance
                    closest_node = node
        
        if closest_node is not None:
            # Create an InsertOperator to add this node to the solution
            return InsertOperator(vehicle_id, closest_node, len(state_data['current_solution'].routes[vehicle_id])), {}
    
    # If all nodes are visited or no capacity is left to visit any node, return None
    return None, {}