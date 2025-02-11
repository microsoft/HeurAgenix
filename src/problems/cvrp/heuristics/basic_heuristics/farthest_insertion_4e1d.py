from src.problems.cvrp.components import Solution, AppendOperator, InsertOperator
import numpy as np

def farthest_insertion_4e1d(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AppendOperator, dict]:
    """ Farthest Insertion heuristic for the CVRP.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "node_num" (int): The total number of nodes in the problem.
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
            - "vehicle_num" (int): The total number of vehicles.
            - "capacity" (int): The capacity for each vehicle and all vehicles share the same value.
            - "depot" (int): The index for depot node.
            - "demands" (numpy.ndarray): The demand of each node.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current set of routes for all vehicles.
            - "unvisited_nodes" (list[int]): Nodes that have not yet been visited by any vehicle.
            - "vehicle_loads" (list[int]): The current load of each vehicle.
            - "vehicle_remaining_capacity" (list[int]): The remaining capacity for each vehicle.
            - "validation_solution" (callable): A function to check whether a new solution is valid.
        get_state_data_function (callable): The function that receives the new solution as input and returns the state dictionary for the new solution, without modifying the original solution.

    Returns:
        AppendOperator: An operator that represents the insertion of a node into the route.
        dict: An empty dictionary since this heuristic does not update algorithm_data.
    """
    distance_matrix = global_data["distance_matrix"]
    depot = global_data["depot"]
    demands = global_data["demands"]
    unvisited_nodes = state_data["unvisited_nodes"]
    vehicle_loads = state_data["vehicle_loads"]
    vehicle_remaining_capacity = state_data["vehicle_remaining_capacity"]
    current_solution = state_data["current_solution"]

    # If all nodes are visited, return None
    if not unvisited_nodes:
        return None, {}

    # Start with the farthest node from the depot
    farthest_node = max(unvisited_nodes, key=lambda node: distance_matrix[depot][node])
    best_insertion = None
    min_cost_increase = float('inf')

    # Try to insert the farthest node into each route at the best position
    for vehicle_id, route in enumerate(current_solution.routes):
        if demands[farthest_node] <= vehicle_remaining_capacity[vehicle_id]:
            # Try every possible position in the route
            for position in range(1, len(route) + 1):
                # Calculate the cost increase if inserting the node at this position
                previous_node = route[position - 1] if position > 0 else depot
                next_node = route[position] if position < len(route) else depot
                cost_increase = (distance_matrix[previous_node][farthest_node] +
                                 distance_matrix[farthest_node][next_node] -
                                 distance_matrix[previous_node][next_node])

                # Update the best insertion if the cost is lower
                if cost_increase < min_cost_increase:
                    min_cost_increase = cost_increase
                    best_insertion = InsertOperator(vehicle_id, farthest_node, position)

    # If a valid insertion is found, return it
    if best_insertion is not None:
        return best_insertion, {}

    # If no valid insertion is found, return None
    return None, {}