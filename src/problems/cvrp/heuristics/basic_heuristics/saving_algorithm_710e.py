from src.problems.cvrp.components import Solution, MergeRoutesOperator

def saving_algorithm_710e(problem_state: dict, algorithm_data: dict, merge_threshold: float = 0.0, **kwargs) -> tuple[MergeRoutesOperator, dict]:
    """ Saving Algorithm heuristic for CVRP.
    Calculates savings for combining two routes into one for each pair of end nodes of existing routes and sorts these savings in descending order. Routes are then merged by the highest savings while considering vehicle capacity constraints.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): The distance matrix between nodes.
            - "vehicle_num" (int): The total number of vehicles.
            - "capacity" (int): The capacity for each vehicle.
            - "current_solution" (Solution): The current set of routes for all vehicles.
            - "vehicle_loads" (list[int]): The current load of each vehicle.
            - "unvisited_nodes" (list[int]): Nodes that have not yet been visited by any vehicle.
        algorithm_data (dict): Not used in this algorithm.
        merge_threshold (float): A hyperparameter to set the minimum saving value to consider for merging routes. Default is 0.0.

    Returns:
        MergeRoutesOperator: The chosen operator to merge two routes if a saving above the threshold is found.
        dict: Empty dictionary as no algorithm_data is updated.
    """
    distance_matrix = problem_state["distance_matrix"]
    vehicle_capacity = problem_state["capacity"]
    depot = problem_state["depot"]
    vehicle_loads = problem_state["vehicle_loads"]
    current_solution = problem_state["current_solution"]

    best_saving = merge_threshold
    best_operator = None

    for i, route1 in enumerate(current_solution.routes):
        if len(route1) <= 1:
            continue
        for j, route2 in enumerate(current_solution.routes):
            if i == j or not len(route2) <= 1:
                continue
            depot_index1 = route1.index(depot)
            rotated_route1 = route1[depot_index1:] + route1[:depot_index1]
            depot_index2 = route2.index(depot)
            rotated_route2 = route1[depot_index2:] + route1[:depot_index2]
            saving = (distance_matrix[rotated_route1[-1]][depot] + distance_matrix[depot][rotated_route2[1]]) - distance_matrix[rotated_route1[-1]][rotated_route2[1]]
            if saving > best_saving and vehicle_loads[i] + vehicle_loads[j] <= vehicle_capacity:
                best_saving = saving
                best_operator = MergeRoutesOperator(source_vehicle_id=i, target_vehicle_id=j)

    if best_operator:
        return best_operator, {}
    else:
        # No valid merge found, or all merges have savings below the threshold
        return None, {}