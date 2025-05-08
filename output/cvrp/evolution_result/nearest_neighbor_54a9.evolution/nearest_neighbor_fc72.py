from src.problems.cvrp.components import *
import numpy as np

def nearest_neighbor_fc72(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """ Enhanced Nearest Neighbor Heuristic for CVRP with capacity utilization and insertion logic.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - distance_matrix (numpy.ndarray): 2D array representing distances between nodes.
            - demands (numpy.ndarray): Array of demands for each node.
            - capacity (int): Capacity of each vehicle.
            - depot (int): Index of the depot node.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current routes for all vehicles.
            - unvisited_nodes (list[int]): Nodes not yet visited by any vehicle.
            - vehicle_remaining_capacity (list[int]): Remaining capacity for each vehicle.
        get_state_data_function (callable): Function to calculate state data for a new solution.
        **kwargs: Hyper-parameters for the heuristic algorithm.
            - capacity_utilization_threshold (float, default=0.9): Threshold to prioritize vehicles with utilization below this level.
            - high_demand_threshold (int, default=6): Threshold for considering the number of unvisited nodes as low.

    Returns:
        InsertOperator or AppendOperator: Operator to modify the current solution.
        dict: Updated algorithm data if any.
    """
    # Extract required data
    distance_matrix = global_data["distance_matrix"]
    demands = global_data["demands"]
    capacity = global_data["capacity"]
    depot = global_data["depot"]

    current_solution = state_data["current_solution"]
    unvisited_nodes = state_data["unvisited_nodes"]
    vehicle_remaining_capacity = state_data["vehicle_remaining_capacity"]

    # Hyper-parameters
    capacity_utilization_threshold = kwargs.get("capacity_utilization_threshold", 0.9)
    high_demand_threshold = kwargs.get("high_demand_threshold", 6)

    # Check if there are no unvisited nodes
    if not unvisited_nodes:
        return None, {}

    # Prioritize nodes with higher demands when unvisited nodes are below the threshold
    if len(unvisited_nodes) <= high_demand_threshold:
        best_cost_increase = float('inf')
        best_node = None
        best_vehicle = None
        best_position = None

        for node in unvisited_nodes:
            for vehicle_id, remaining_capacity in enumerate(vehicle_remaining_capacity):
                if demands[node] > remaining_capacity:
                    continue

                # Calculate current capacity utilization
                current_utilization = (capacity - remaining_capacity) / capacity

                # Skip vehicles above utilization threshold
                if current_utilization >= capacity_utilization_threshold:
                    continue

                # Try inserting the node at all possible positions in the vehicle's route
                route = current_solution.routes[vehicle_id]
                for position in range(len(route) + 1):
                    prev_node = depot if position == 0 else route[position - 1]
                    next_node = depot if position == len(route) else route[position]

                    cost_increase = (
                        distance_matrix[prev_node][node] +
                        distance_matrix[node][next_node] -
                        distance_matrix[prev_node][next_node]
                    )

                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_node = node
                        best_vehicle = vehicle_id
                        best_position = position

        if best_node is not None and best_vehicle is not None and best_position is not None:
            # Consider appending if it results in better utilization
            if best_position == len(current_solution.routes[best_vehicle]):
                return AppendOperator(best_vehicle, best_node), {}
            else:
                return InsertOperator(best_vehicle, best_node, best_position), {}

    # Default nearest neighbor logic
    best_cost_increase = float('inf')
    best_node = None
    best_vehicle = None
    best_position = None

    for node in unvisited_nodes:
        for vehicle_id, remaining_capacity in enumerate(vehicle_remaining_capacity):
            if demands[node] > remaining_capacity:
                continue

            # Calculate current capacity utilization
            current_utilization = (capacity - remaining_capacity) / capacity

            # Skip vehicles above utilization threshold
            if current_utilization >= capacity_utilization_threshold:
                continue

            # Try inserting the node at all possible positions in the vehicle's route
            route = current_solution.routes[vehicle_id]
            for position in range(len(route) + 1):
                prev_node = depot if position == 0 else route[position - 1]
                next_node = depot if position == len(route) else route[position]

                cost_increase = (
                    distance_matrix[prev_node][node] +
                    distance_matrix[node][next_node] -
                    distance_matrix[prev_node][next_node]
                )

                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_node = node
                    best_vehicle = vehicle_id
                    best_position = position

    if best_node is not None and best_vehicle is not None and best_position is not None:
        # Consider appending if it results in better utilization
        if best_position == len(current_solution.routes[best_vehicle]):
            return AppendOperator(best_vehicle, best_node), {}
        else:
            return InsertOperator(best_vehicle, best_node, best_position), {}

    # If no valid insertion was found
    return None, {}