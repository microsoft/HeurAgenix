from src.problems.cvrp.components import Solution, InsertOperator

def greedy_f4c4(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Constructive nearest-from-depot assignment for CVRP. Each call scans vehicles in ascending ID and selects the first vehicle with remaining capacity to perform a single insertion: among all unvisited nodes whose demand fits that vehicle, pick the node with minimum distance from the depot and append it to the end of that vehicle’s route.

    Key characteristics:
    - “Best” is evaluated only by depot→node distance for the chosen vehicle; incremental route cost from the current tail is not considered.
    - Single-step extension per call; no lookahead, no route merging, no closure to the depot.
    - Capacity feasibility enforced nodewise (demands[node] ≤ remaining capacity); no global balancing.
    - Vehicle priority is first-fit by ID and serves as a tie-breaker across vehicles.
    - Positioning is end-of-route only; existing route order remains unchanged.
    - Depot is assumed to be ID 0 for distance evaluation.

    Inputs used: distance_matrix, demands, vehicle_num, capacity, unvisited_nodes, vehicle_remaining_capacity, current_solution (for route length only).
    Time complexity per call: O(vehicle_num + |unvisited_nodes|); constant extra memory.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): The distances between nodes.
            - "demands" (numpy.ndarray): The demand of each node.
            - "vehicle_num" (int): The total number of vehicles.
            - "capacity" (int): The capacity for each vehicle.
            - "unvisited_nodes" (list[int]): Nodes that have not yet been visited.
            - "vehicle_remaining_capacity" (list[int]): The remaining capacity for each vehicle.

    Returns:
        InsertOperator: The operator that adds a node to a vehicle's route.
        dict: Empty dictionary as the algorithm does not update any algorithm-specific data.
    """
    
    distance_matrix = problem_state['distance_matrix']
    demands = problem_state['demands']
    vehicle_num = problem_state['vehicle_num']
    vehicle_capacity = problem_state['capacity']
    
    unvisited_nodes = problem_state['unvisited_nodes']
    vehicle_remaining_capacity = problem_state['vehicle_remaining_capacity']
    
    # Start with the first unvisited node and the first vehicle
    for vehicle_id in range(vehicle_num):
        if not unvisited_nodes or vehicle_remaining_capacity[vehicle_id] <= 0:
            continue  # Skip to the next vehicle if there are no nodes left or no capacity.
        
        # Initialize a temporary variable to find the closest unvisited node
        closest_node = None
        closest_distance = float('inf')
        
        last_visited_node = 0  # Assuming the depot ID is 0
        for node in unvisited_nodes:
            if demands[node] <= vehicle_remaining_capacity[vehicle_id]:
                distance = distance_matrix[last_visited_node][node]
                if distance < closest_distance:
                    closest_distance = distance
                    closest_node = node
        
        if closest_node is not None:
            # Create an InsertOperator to add this node to the solution
            return InsertOperator(vehicle_id, closest_node, len(problem_state['current_solution'].routes[vehicle_id])), {}
    
    # If all nodes are visited or no capacity is left to visit any node, return None
    return None, {}