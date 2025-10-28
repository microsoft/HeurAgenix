from src.problems.cvrp.components import *

def nearest_neighbor_99ba(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Nearest Neighbor heuristic for CVRP.
    This algorithm iterates over each vehicle, starting from the depot. For each vehicle, it finds the nearest unvisited node and appends it to the route, respecting the vehicle's capacity constraints.
    This process repeats until no further nodes can be visited without exceeding the vehicle's capacity or all nodes have been visited.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
            - "node_num" (int): The total number of nodes in the problem.
            - "depot" (int): The index for depot node.
            - "demands" (numpy.ndarray): The demand of each node.
            - "capacity" (int): The capacity for each vehicle.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "unvisited_nodes" (list[int]): Nodes that have not yet been visited by any vehicle.
            - "vehicle_remaining_capacity" (list[int]): The remaining capacity for each vehicle.
            - "current_solution" (Solution): The current set of routes for all vehicles.

    Returns:
        InsertOperator: The operator to insert the nearest neighbor node into the vehicle's route.
        dict: Empty dictionary as this algorithm does not update the algorithm data.
    """
    distance_matrix = global_data['distance_matrix']
    demands = global_data['demands']
    capacity = global_data['capacity']
    depot = global_data['depot']

    unvisited_nodes = state_data['unvisited_nodes']
    vehicle_remaining_capacity = state_data['vehicle_remaining_capacity']
    current_solution = state_data['current_solution'].routes

    # Iterate over each vehicle
    for vehicle_id, remaining_capacity in enumerate(vehicle_remaining_capacity):
        if not unvisited_nodes or remaining_capacity <= 0:
            # If there are no unvisited nodes or the vehicle has no remaining capacity, continue to the next vehicle
            continue

        last_visited = depot if not current_solution[vehicle_id] else current_solution[vehicle_id][-1]
        nearest_node = None
        min_distance = float('inf')

        # Find the nearest unvisited node that doesn't exceed the vehicle's capacity
        for node in unvisited_nodes:
            if demands[node] <= remaining_capacity and distance_matrix[last_visited][node] < min_distance:
                nearest_node = node
                min_distance = distance_matrix[last_visited][node]

        if nearest_node is not None:
            # If a nearest node is found, create an operator to insert the node into the current vehicle's route
            return InsertOperator(vehicle_id, nearest_node, len(current_solution[vehicle_id])), {}

    # If all vehicles have no remaining capacity or all nodes are visited, return None
    return None, {}