from src.problems.cvrp.components import Solution, InsertOperator

def farthest_insertion_ce2b(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Farthest Insertion heuristic for the CVRP.

    This algorithm selects the unvisited node that is farthest from the depot and inserts it into a route that minimizes the 
    increase in total distance, while also considering vehicle capacity constraints. The heuristic continues until no further 
    nodes can be assigned without violating capacity constraints or all nodes are assigned.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): Distance matrix representing the distances between nodes.
            - "demands" (numpy.ndarray): The demand of each node.
            - "vehicle_num" (int): The total number of vehicles.
            - "capacity" (int): The capacity for each vehicle (assumed to be the same for all vehicles).
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "unvisited_nodes" (list[int]): Nodes that have not yet been visited by any vehicle.
            - "vehicle_remaining_capacity" (list[int]): The remaining capacity for each vehicle.
            - "current_solution" (Solution): The current set of routes for all vehicles.

    Returns:
        InsertOperator: The operator that will insert the farthest node into the best position in a route.
        dict: Empty dictionary as no additional algorithm data is updated by this heuristic.
    """
    # Retrieve necessary data from global_data
    distance_matrix = global_data['distance_matrix']
    demands = global_data['demands']
    vehicle_num = global_data['vehicle_num']
    capacity = global_data['capacity']
    
    # Retrieve necessary data from state_data
    unvisited_nodes = state_data['unvisited_nodes']
    vehicle_remaining_capacity = state_data['vehicle_remaining_capacity']
    current_solution = state_data['current_solution']
    
    # Check if there are unvisited nodes left
    if not unvisited_nodes:
        return None, {}  # No unvisited nodes to insert, return None

    # Find the farthest unvisited node from the depot
    depot = 0  # Assuming the depot's index is 0
    farthest_node = max(unvisited_nodes, key=lambda node: distance_matrix[depot][node])
    demand_of_farthest = demands[farthest_node]
    
    # Initialize variables to keep track of the best insertion
    best_cost_increase = float('inf')
    best_vehicle_id = None
    best_position = None
    
    # Iterate over each vehicle's route to find the best position for insertion
    for vehicle_id in range(vehicle_num):
        if vehicle_remaining_capacity[vehicle_id] >= demand_of_farthest:
            route = current_solution.routes[vehicle_id]
            
            # Check insertion cost for each possible position in the route
            for position in range(len(route) + 1):
                # Calculate the cost increase of this insertion
                if position == 0:  # Insertion at the start of the route
                    before_node = depot
                    after_node = route[0] if route else depot
                elif position == len(route):  # Insertion at the end of the route
                    before_node = route[-1]
                    after_node = depot
                else:  # Insertion in the middle of the route
                    before_node = route[position - 1]
                    after_node = route[position]
                
                # Calculate the cost increase if the farthest node was inserted here
                cost_increase = (distance_matrix[before_node][farthest_node] +
                                 distance_matrix[farthest_node][after_node] -
                                 distance_matrix[before_node][after_node])
                
                # Update the best insertion if this is the lowest cost increase found so far
                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_vehicle_id = vehicle_id
                    best_position = position
    
    # If no feasible insertion was found, return None
    if best_vehicle_id is None:
        return None, {}
    
    # Create the operator to perform the insertion
    insert_operator = InsertOperator(vehicle_id=best_vehicle_id, node=farthest_node, position=best_position)

    
    return insert_operator, {}