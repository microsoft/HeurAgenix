from src.problems.vrptw.components import *
import numpy as np

def nearest_neighbor_6b97(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AppendOperator, dict]:
    """Constructs a route by repeatedly visiting the nearest unvisited customer that can be feasibly visited within the vehicle's capacity and time window constraints.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
            - "capacity" (int): The capacity for each vehicle and all vehicles share the same value.
            - "depot" (int): The index for the depot node.
            - "time_windows" (N * 2 numpy.ndarray): The ready and due time for each node.
            - "demands" (numpy.ndarray): The demand of each node.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current set of routes for all vehicles.
            - "unvisited_nodes" (list[int]): Nodes that have not yet been visited by any vehicle.
            - "vehicle_remaining_capacity" (list[int]): The remaining capacity for each vehicle.
            - "detailed_time" (list[list[list]]): Detailed time-related information for each vehicle and its tasks.
            - "last_visited" (list[int]): The last visited node for each vehicle.
            - "validation_single_route" (callable): Function to check whether the single route is valid, including capacity and time window check.
    Returns:
        The nearest neighbor operator instance that can be applied to the current solution to improve it.
        An empty dict as no algorithm-specific data is updated here.
    """
    distance_matrix = global_data["distance_matrix"]
    depot = global_data["depot"]
    time_windows = global_data["time_windows"]
    demands = global_data["demands"]
    current_solution = state_data["current_solution"]
    unvisited_nodes = state_data["unvisited_nodes"]
    vehicle_remaining_capacity = state_data["vehicle_remaining_capacity"]
    detailed_time = state_data["detailed_time"]
    last_visited = state_data["last_visited"]
    validation_single_route = state_data["validation_single_route"]

    min_distance = float('inf')
    best_vehicle_id = None
    best_node = None

    for vehicle_id, route in enumerate(current_solution.routes):
        current_node = last_visited[vehicle_id]
        for node in unvisited_nodes:
            # Check capacity constraint
            if demands[node] > vehicle_remaining_capacity[vehicle_id]:
                continue

            # Calculate the potential arrival time at this node
            arrival_time = detailed_time[vehicle_id][-1][3] + distance_matrix[current_node][node]  # Finish time + travel time

            # Check time window constraint
            if arrival_time > time_windows[node][1]:  # Due time
                continue

            # Construct a potential new route with this node included
            potential_route = route[:-1] + [node, depot]  # Insert node and append depot

            # Validate the new route
            if validation_single_route(potential_route):
                # Check if this node is the closest feasible node
                if distance_matrix[current_node][node] < min_distance:
                    min_distance = distance_matrix[current_node][node]
                    best_vehicle_id = vehicle_id
                    best_node = node

    if best_node is not None:
        # Apply the AppendOperator to add the best node to the best vehicle's route
        append_operator = AppendOperator(best_vehicle_id, best_node)
        return append_operator, {}

    return None, {}