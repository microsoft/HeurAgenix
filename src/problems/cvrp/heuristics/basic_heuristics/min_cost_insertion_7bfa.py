from src.problems.cvrp.components import Solution, InsertOperator

def min_cost_insertion_7bfa(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Min-Cost Insertion heuristic for the CVRP.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - distance_matrix (numpy.ndarray): A matrix of distances between nodes.
            - depot (int): The index for depot node.
            - demands (list[int]): The demand of each node.
            - vehicle_num (int): The total number of vehicles.
            - capacity (int): The capacity for each vehicle.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): The current set of routes for all vehicles.
            - unvisited_nodes (list[int]): Nodes that have not yet been visited by any vehicle.
            - vehicle_remaining_capacity (list[int]): The remaining capacity for each vehicle.

    Returns:
        InsertOperator: The operator to insert a node into the route at the position that minimizes the cost increase.
        dict: An empty dictionary since this algorithm doesn't update the algorithm data.
    """
    distance_matrix = global_data['distance_matrix']
    demands = global_data['demands']
    depot = global_data['depot']
    unvisited_nodes = state_data['unvisited_nodes']
    vehicle_remaining_capacity = state_data['vehicle_remaining_capacity']
    current_solution = state_data['current_solution']

    # Find the best insertion for the next unvisited node
    best_cost_increase = float('inf')
    best_insert_op = None
    for node in unvisited_nodes:
        node_demand = demands[node]
        for vehicle_id in range(len(current_solution.routes)):
            if vehicle_remaining_capacity[vehicle_id] >= node_demand:
                for insert_pos in range(len(current_solution.routes[vehicle_id]) + 1):
                    # Calculate the cost increase if we insert the node at this position
                    before_node = current_solution.routes[vehicle_id][insert_pos - 1] if insert_pos > 0 else depot
                    after_node = current_solution.routes[vehicle_id][insert_pos] if insert_pos < len(current_solution.routes[vehicle_id]) else depot
                    cost_increase = (distance_matrix[before_node][node] +
                                     distance_matrix[node][after_node] -
                                     distance_matrix[before_node][after_node])
                    # If this position is better than the current best, update the best
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_insert_op = InsertOperator(vehicle_id, node, insert_pos)

    # If no valid insertion could be found, return None
    if best_insert_op is None:
        return None, {}

    # Return the best insertion operator found
    return best_insert_op, {}