from src.problems.cvrp.components import Solution, AppendOperator, InsertOperator
import numpy as np

def min_cost_insertion_048f(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """ Min-Cost Insertion heuristic for the CVRP.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "node_num" (int): Total number of nodes.
            - "distance_matrix" (numpy.ndarray): 2D array representing distances between nodes.
            - "vehicle_num" (int): Total number of vehicles.
            - "capacity" (int): Capacity for each vehicle.
            - "depot" (int): Index for depot node.
            - "demands" (numpy.ndarray): Demand of each node.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): Current set of routes.
            - "unvisited_nodes" (list[int]): Nodes not yet visited.
            - "vehicle_loads" (list[int]): Current load of each vehicle.
            - "vehicle_remaining_capacity" (list[int]): Remaining capacity for each vehicle.
        get_state_data_function (callable): Function to get state data for a new solution.
        kwargs: Additional hyper-parameters for the algorithm. Default values should be set here if needed.

    Returns:
        An InsertOperator for inserting the node at the optimal position.
        An updated algorithm data dictionary.
    """

    # Extract necessary data
    distance_matrix = global_data["distance_matrix"]
    depot = global_data["depot"]
    demands = global_data["demands"]
    
    current_solution = state_data["current_solution"]
    unvisited_nodes = state_data["unvisited_nodes"]
    vehicle_loads = state_data["vehicle_loads"]
    vehicle_remaining_capacity = state_data["vehicle_remaining_capacity"]

    # Initialize variables to track the best insertion
    best_increase = float('inf')
    best_operator = None

    # Iterate over all unvisited nodes to find the best insertion point
    for node in unvisited_nodes:
        node_demand = demands[node]

        # Check each vehicle's route for possible insertion points
        for vehicle_id, route in enumerate(current_solution.routes):
            if vehicle_remaining_capacity[vehicle_id] < node_demand:
                continue

            # Iterate over all possible positions to insert the node
            for position in range(1, len(route) + 1):
                # Calculate cost increase for inserting the node
                prev_node = depot if position == 1 else route[position - 1]
                next_node = route[position] if position < len(route) else route[0]
                increase = (distance_matrix[prev_node][node] +
                            distance_matrix[node][next_node] -
                            distance_matrix[prev_node][next_node])

                # Check if this is the best insertion found
                if increase < best_increase:
                    best_increase = increase
                    best_operator = InsertOperator(vehicle_id, node, position)

    # If no valid insertion was found, return None
    if best_operator is None:
        return None, {}

    # Return the best insertion operator found
    return best_operator, {}