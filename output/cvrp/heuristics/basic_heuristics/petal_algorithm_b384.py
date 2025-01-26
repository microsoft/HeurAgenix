from src.problems.cvrp.components import Solution, InsertOperator
import math

def petal_algorithm_b384(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Petal construction heuristic algorithm for the CVRP. This heuristic attempts to build feasible routes (petals) around the depot and combine them into a set of vehicle routes.
    Each petal starts and ends at the depot, creating a loop that can serve as an individual route for a vehicle.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
            - "demands" (numpy.ndarray): The demand of each node.
            - "capacity" (int): The capacity for each vehicle.
            - "vehicle_num" (int): The total number of vehicles.
            - "node_num" (int): The total number of nodes in the problem.

        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current set of routes for all vehicles.
            - "unvisited_nodes" (list[int]): Nodes that have not yet been visited by any vehicle.
            - "vehicle_remaining_capacity" (list[int]): The remaining capacity for each vehicle.

    Returns:
        InsertOperator: The operator that adds a node to the route of a vehicle.
        dict: Updated algorithm data, empty in this case as this heuristic doesn't utilize it.
    """

    distance_matrix = global_data["distance_matrix"]
    demands = global_data["demands"]
    vehicle_capacity = global_data["capacity"]
    vehicle_remaining_capacity = state_data["vehicle_remaining_capacity"].copy()
    current_solution = state_data["current_solution"].routes
    unvisited_nodes = state_data["unvisited_nodes"].copy()

    if not unvisited_nodes:
        # No more nodes to visit, return None.
        return None, {}

    # Sort nodes based on their polar angle with respect to the depot
    depot = 0
    sorted_nodes = sorted(unvisited_nodes, key=lambda node: polar_angle(distance_matrix, depot, node))

    # Create petals
    petals = []
    for node in sorted_nodes:
        if demands[node] <= vehicle_capacity:
            petals.append([node])

    # Try to fit petals into existing vehicle routes
    for petal in petals:
        for vehicle_id, route in enumerate(current_solution):
            if route_fits(petal, vehicle_id, vehicle_remaining_capacity, demands, vehicle_capacity):
                # If the petal fits, assign it to the vehicle and return the operator
                return InsertOperator(vehicle_id=vehicle_id, node=petal[0], position=len(route)), {}

    # If no petals fit, return None
    return None, {}

def polar_angle(distance_matrix, depot, node):
    """Calculate the polar angle of a node with respect to the depot for sorting."""
    y_diff = distance_matrix[depot][node] - distance_matrix[depot][0]
    x_diff = distance_matrix[node][depot] - distance_matrix[0][depot]
    return math.atan2(y_diff, x_diff)

def route_fits(petal, vehicle_id, vehicle_remaining_capacity, demands, vehicle_capacity):
    """Check if a petal can fit into the current route of a vehicle."""
    # Calculate the total demand of the petal
    petal_demand = sum(demands[node] for node in petal)
    # Check if adding the petal would exceed the vehicle's capacity
    return petal_demand <= vehicle_remaining_capacity[vehicle_id]