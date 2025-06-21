from src.problems.cvrp.components import AppendOperator
import random

def random_bfdc(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AppendOperator, dict]:
    """Random heuristic for CVRP.
    This heuristic selects an unvisited node at random and appends it to a vehicle's route where it does not violate the capacity constraint.
    This process is repeated for each vehicle until all nodes have been included in a route.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "demands" (numpy.ndarray): The demand of each node.
            - "vehicle_num" (int): The total number of vehicles.
            - "capacity" (int): The capacity for each vehicle.
            - "unvisited_nodes" (list[int]): Nodes that have not yet been visited by any vehicle.
            - "vehicle_remaining_capacity" (list[int]): The remaining capacity for each vehicle.
            - "current_solution" (Solution): The current set of routes for all vehicles.

    Returns:
        AppendOperator: The operator to append a node to a vehicle's route.
        dict: Empty dictionary as no algorithm data needs to be updated.
    """
    unvisited_nodes = problem_state['unvisited_nodes']
    vehicle_remaining_capacity = problem_state['vehicle_remaining_capacity']
    current_solution = problem_state['current_solution']
    demands = problem_state['demands']
    vehicle_num = problem_state['vehicle_num']
    capacity = problem_state['capacity']

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