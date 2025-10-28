from src.problems.cvrp.components import *

def variable_neighborhood_search_614b(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[BaseOperator, dict]:
    """ Variable Neighborhood Search heuristic algorithm for CVRP.
    This function performs a Variable Neighborhood Search by systematically changing the neighborhood structure within a local search algorithm to escape local optima and search for better solutions.
    It uses a series of pre-defined operators to create new neighborhoods and improve upon the current solution.

    Args:
        global_data (dict): Global data containing problem parameters.
            - "distance_matrix" (numpy.ndarray): Distance between nodes.
            - "vehicle_num" (int): Number of vehicles.
            - "capacity" (int): Capacity of each vehicle.
            - "depot" (int): Index of the depot node.
        state_data (dict): State data containing the current state of the solution.
            - "current_solution" (Solution): The current set of routes for all vehicles.
            - "vehicle_loads" (list[int]): The current load of each vehicle.
            - "unvisited_nodes" (list[int]): Nodes that have not yet been visited by any vehicle.
        algorithm_data (dict): Algorithm-specific data, not used in this heuristic.
        **kwargs: Hyperparameters for the search, such as the neighborhood size.

    Returns:
        BaseOperator: The operator that modifies the current solution, or None if no improvement is possible.
        dict: Empty dictionary as this function does not update algorithm_data.
    """

    # Retrieve necessary data from state_data
    current_solution = state_data.get('current_solution')
    vehicle_loads = state_data.get('vehicle_loads')
    unvisited_nodes = state_data.get('unvisited_nodes')

    # Retrieve necessary data from global_data
    depot = global_data.get('depot')
    capacity = global_data.get('capacity')
    vehicle_num = global_data.get('vehicle_num')
    distance_matrix = global_data.get('distance_matrix')

    # Define hyperparameters (with default values)
    neighborhood_size = kwargs.get('neighborhood_size', 10)

    # Initialize variables for the best modification found
    best_operator = None
    best_cost_saving = float('inf')

    # Iterate over all possible neighborhoods defined by the neighborhood size
    for vehicle_id in range(vehicle_num):
        for node_index in range(len(unvisited_nodes)):
            node = unvisited_nodes[node_index]
            # Check if adding this node to the route exceeds the vehicle's capacity
            if vehicle_loads[vehicle_id] + global_data['demands'][node] <= capacity:
                for position in range(len(current_solution.routes[vehicle_id]) + 1):
                    # Calculate the cost of inserting the node at the current position
                    if position == 0:
                        before_node = depot
                    else:
                        before_node = current_solution.routes[vehicle_id][position - 1]
                    if position == len(current_solution.routes[vehicle_id]):
                        after_node = depot
                    else:
                        after_node = current_solution.routes[vehicle_id][position]
                    cost_to_add = distance_matrix[before_node][node] + distance_matrix[node][after_node] - distance_matrix[before_node][after_node]

                    # If the insertion leads to a cost saving, and is better than the previous best saving, store it
                    if cost_to_add < best_cost_saving:
                        best_cost_saving = cost_to_add
                        best_operator = InsertOperator(vehicle_id=vehicle_id, node=node, position=position)

    # Return the best operator found, or None if no improving operator was found
    return best_operator or None, {}