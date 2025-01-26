from src.problems.cvrp.components import Solution, MergeRoutesOperator

def saving_algorithm_710e(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, merge_threshold: float = 0.0, **kwargs) -> tuple[MergeRoutesOperator, dict]:
    """ Saving Algorithm heuristic for CVRP.
    Calculates savings for combining two routes into one for each pair of end nodes of existing routes and sorts these savings in descending order. Routes are then merged by the highest savings while considering vehicle capacity constraints.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): The distance matrix between nodes.
            - "vehicle_num" (int): The total number of vehicles.
            - "capacity" (int): The capacity for each vehicle.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current set of routes for all vehicles.
            - "vehicle_loads" (list[int]): The current load of each vehicle.
            - "unvisited_nodes" (list[int]): Nodes that have not yet been visited by any vehicle.
        algorithm_data (dict): Not used in this algorithm.
        merge_threshold (float): A hyperparameter to set the minimum saving value to consider for merging routes. Default is 0.0.

    Returns:
        MergeRoutesOperator: The chosen operator to merge two routes if a saving above the threshold is found.
        dict: Empty dictionary as no algorithm_data is updated.
    """
    distance_matrix = global_data["distance_matrix"]
    vehicle_capacity = global_data["capacity"]
    depot = global_data["depot"]
    vehicle_loads = state_data["vehicle_loads"]
    current_solution = state_data["current_solution"]

    best_saving = merge_threshold
    best_operator = None

    for i, route1 in enumerate(current_solution.routes):
        if not route1:
            continue
        for j, route2 in enumerate(current_solution.routes):
            if i == j or not route2:
                continue
            saving = (distance_matrix[route1[-1]][depot] + distance_matrix[depot][route2[0]]) - distance_matrix[route1[-1]][route2[0]]
            if saving > best_saving and vehicle_loads[i] + vehicle_loads[j] <= vehicle_capacity:
                best_saving = saving
                best_operator = MergeRoutesOperator(source_vehicle_id=i, target_vehicle_id=j)

    if best_operator:
        return best_operator, {}
    else:
        # No valid merge found, or all merges have savings below the threshold
        return None, {}