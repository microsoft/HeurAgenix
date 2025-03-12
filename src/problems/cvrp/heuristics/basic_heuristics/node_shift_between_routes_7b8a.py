from src.problems.cvrp.components import *

def node_shift_between_routes_7b8a(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[RelocateOperator, dict]:
    """Node Shift Between Routes heuristic for CVRP.
    This heuristic attempts to move a node from one route to another route, aiming to reduce the total distance or improve load balance.
    It considers the capacity constraints and ensures that the move is beneficial before applying it.

    Args:
        global_data (dict): Contains the global information about the problem.
            - "distance_matrix" (numpy.ndarray): The matrix of distances between nodes.
            - "capacity" (int): The capacity for each vehicle.
            
        state_data (dict): Contains the current state of the solution.
            - "current_solution" (Solution): The current set of routes.
            - "vehicle_loads" (list[int]): The current load of each vehicle.
            - "unvisited_nodes" (list[int]): Nodes that have not yet been visited.
            
        algorithm_data (dict): Contains algorithm-specific data. Not used in this heuristic.

    Returns:
        (RelocateOperator, dict): The operator to shift a node between routes and an empty dictionary, as the heuristic does not update algorithm_data.
    """
    # Extract required data from global_data
    distance_matrix = global_data["distance_matrix"]
    capacity = global_data["capacity"]
    depot = global_data["depot"]
    
    # Extract required data from state_data
    current_solution = state_data["current_solution"]
    vehicle_loads = state_data["vehicle_loads"]
    # Find the best position to insert the node in the target route
    best_position = None
    best_cost_reduction = 0
    # Find the best node shift between routes
    for source_vehicle_id, source_route in enumerate(current_solution.routes):
        for source_position, node in enumerate(source_route):
            # Skip if no nodes to shift
            if not source_route or node == depot:
                continue
            
            # Calculate the load after removing the node
            new_load_source = vehicle_loads[source_vehicle_id] - global_data["demands"][node]
            if new_load_source < 0:
                continue  # Skip if moving the node violates source vehicle's capacity
            
            # Check each target route to find the best shift
            for target_vehicle_id, target_route in enumerate(current_solution.routes):
                # Do not consider the same route
                if source_vehicle_id == target_vehicle_id:
                    continue
                
                # Calculate the load after adding the node to the target vehicle
                new_load_target = vehicle_loads[target_vehicle_id] + global_data["demands"][node]
                if new_load_target > capacity:
                    continue  # Skip if moving the node violates target vehicle's capacity
                for target_position in range(len(target_route) + 1):
                        # Calculate the cost difference if the node is inserted at the target position
                        source_previous_node = source_route[(source_position - 1) % len(source_route)]
                        source_next_node = source_route[(source_position + 1) % len(source_route)]
                        target_previous_node = target_route[(target_position - 1) % len(target_route)]
                        target_next_node = target_route[target_position % len(target_route)]

                        cost_increase = (
                            -distance_matrix[source_previous_node][node]
                            -distance_matrix[node][source_next_node]
                            +distance_matrix[source_previous_node][source_next_node]
                            +distance_matrix[target_previous_node][node]
                            +distance_matrix[node][target_next_node]
                            -distance_matrix[target_previous_node][target_next_node]
                        )
                        cost_reduction = -cost_increase

                        # Update best shift if this shift is better
                        if cost_reduction > best_cost_reduction and current_solution.routes[source_vehicle_id][source_position] != depot:
                            best_source_vehicle_id = source_vehicle_id
                            best_source_position = source_position
                            best_target_vehicle_id = target_vehicle_id
                            best_target_position = target_position
                            best_cost_reduction = cost_reduction
                        
                
    # If a beneficial shift is found, return the corresponding operator
    if best_cost_reduction > 0:
        return RelocateOperator(
            source_vehicle_id=best_source_vehicle_id,
            source_position=best_source_position,
            target_vehicle_id=best_target_vehicle_id,
            target_position=best_target_position
        ), {}
    
    # If no beneficial shift is found, return None
    return None, {}