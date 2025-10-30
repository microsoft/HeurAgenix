from src.problems.cvrp.components import *

def nearest_neighbor_99ba(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Greedy nearest-neighbor with capacity filter for open CVRP routes. Vehicles are processed in ID order. For the current vehicle, the “last” node is the depot if the route is empty; otherwise it is the route’s final customer. Among all unvisited customers with demand ≤ the vehicle’s remaining capacity, select the one with minimal distance from this last node and append it to the end of that vehicle’s route (InsertOperator at position len(route)). Early exit after the first feasible append: no global competition across vehicles. Consequently, within a vehicle it chooses the best (nearest) candidate, but across vehicles it is “first feasible” rather than globally optimal. Tie-breaking favors the first node encountered at the minimal distance. Does not consider closing to the depot, multi-position insertions, or any local route improvements; pure constructive, adding at most one node per call. Stateless (does not update algorithm_data). Works with asymmetric distance matrices. Time complexity per call: worst-case O(V·U), typically O(U) until a feasible vehicle is found. Capacity is enforced via vehicle_remaining_capacity only.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
            - "node_num" (int): The total number of nodes in the problem.
            - "depot" (int): The index for depot node.
            - "demands" (numpy.ndarray): The demand of each node.
            - "capacity" (int): The capacity for each vehicle.
            - "unvisited_nodes" (list[int]): Nodes that have not yet been visited by any vehicle.
            - "vehicle_remaining_capacity" (list[int]): The remaining capacity for each vehicle.
            - "current_solution" (Solution): The current set of routes for all vehicles.

    Returns:
        InsertOperator: The operator to insert the nearest neighbor node into the vehicle's route.
        dict: Empty dictionary as this algorithm does not update the algorithm data.
    """
    distance_matrix = problem_state['distance_matrix']
    demands = problem_state['demands']
    capacity = problem_state['capacity']
    depot = problem_state['depot']

    unvisited_nodes = problem_state['unvisited_nodes']
    vehicle_remaining_capacity = problem_state['vehicle_remaining_capacity']
    current_solution = problem_state['current_solution'].routes

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