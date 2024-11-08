from src.problems.cvrp.components import AppendOperator
import random

def random_bfdc(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[AppendOperator, dict]:
    """Random heuristic for CVRP.
    This heuristic selects an unvisited node at random and appends it to a vehicle's route where it does not violate the capacity constraint.
    This process is repeated for each vehicle until all nodes have been included in a route.

    Args:
        global_data (dict): The global data dict containing the global data. 
            - "demands" (numpy.ndarray): The demand of each node.
            - "vehicle_num" (int): The total number of vehicles.
            - "capacity" (int): The capacity for each vehicle.
        state_data (dict): The state dictionary containing the current state information.
            - "unvisited_nodes" (list[int]): Nodes that have not yet been visited by any vehicle.
            - "vehicle_remaining_capacity" (list[int]): The remaining capacity for each vehicle.
            - "current_solution" (Solution): The current set of routes for all vehicles.

    Returns:
        AppendOperator: The operator to append a node to a vehicle's route.
        dict: Empty dictionary as no algorithm data needs to be updated.
    """
    unvisited_nodes = state_data['unvisited_nodes']
    vehicle_remaining_capacity = state_data['vehicle_remaining_capacity']
    current_solution = state_data['current_solution']
    demands = global_data['demands']
    vehicle_num = global_data['vehicle_num']
    capacity = global_data['capacity']

    # Check if we have any unvisited nodes left
    if not unvisited_nodes:
        return None, {}

    # Randomly select an unvisited node
    node_to_append = random.choice(unvisited_nodes)

    # Find a vehicle that can accommodate the node
    for _ in range(vehicle_num):
        vehicle_id = random.randrange(vehicle_num)
        if vehicle_remaining_capacity[vehicle_id] >= demands[node_to_append]:
            # Create and return the append operator
            operator = AppendOperator(vehicle_id=vehicle_id, node=node_to_append)
            return operator, {}

    # If we reach here, no vehicle can accommodate the node (should not happen if vehicles start empty and capacities are correct)
    return None, {}