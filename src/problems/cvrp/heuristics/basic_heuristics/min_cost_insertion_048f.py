from src.problems.cvrp.components import Solution, AppendOperator, InsertOperator
import numpy as np

def min_cost_insertion_048f(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertOperator, dict]:
    """ Min-Cost Insertion heuristic for the CVRP.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "node_num" (int): Total number of nodes.
            - "distance_matrix" (numpy.ndarray): 2D array representing distances between nodes.
            - "vehicle_num" (int): Total number of vehicles.
            - "capacity" (int): Capacity for each vehicle.
            - "depot" (int): Index for depot node.
            - "demands" (numpy.ndarray): Demand of each node.
            - "current_solution" (Solution): Current set of routes.
            - "unvisited_nodes" (list[int]): Nodes not yet visited.
            - "vehicle_loads" (list[int]): Current load of each vehicle.
            - "vehicle_remaining_capacity" (list[int]): Remaining capacity for each vehicle.
        kwargs: Additional hyper-parameters for the algorithm. Default values should be set here if needed.

    Returns:
        An InsertOperator for inserting the node at the optimal position.
        An updated algorithm data dictionary.
    """

    # Extract necessary data
    distance_matrix = problem_state["distance_matrix"]
    depot = problem_state["depot"]
    demands = problem_state["demands"]
    
    current_solution = problem_state["current_solution"]
    unvisited_nodes = problem_state["unvisited_nodes"]
    vehicle_loads = problem_state["vehicle_loads"]
    vehicle_remaining_capacity = problem_state["vehicle_remaining_capacity"]

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