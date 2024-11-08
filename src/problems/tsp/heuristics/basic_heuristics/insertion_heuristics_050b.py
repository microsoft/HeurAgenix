from src.problems.tsp.components import *

def insertion_heuristics_050b(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, insertion_strategy: str = 'cheapest', **kwargs) -> tuple[InsertOperator, dict]:
    """ Insertion heuristics algorithm for the TSP problem. Builds a tour by incrementally adding cities to an existing sub-tour.
    This function supports 'cheapest', 'farthest', and 'nearest' insertion strategies.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - distance_matrix (numpy.ndarray): A 2D array representing the distances between nodes.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): An instance of the Solution class representing the current solution.
            - unvisited_nodes (list[int]): A list of integers representing the IDs of nodes that have not yet been visited.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. This algorithm does not use algorithm_data.
        insertion_strategy (str): The strategy to use for insertion. Defaults to 'cheapest'. Other valid values are 'farthest' and 'nearest'.

    Returns:
        InsertOperator: The operator to modify the current solution.
        dict: Empty dictionary as this algorithm does not update algorithm_data.
    """
    # Extract necessary data from global_data and state_data
    distance_matrix = global_data['distance_matrix']
    current_solution = state_data['current_solution']
    unvisited_nodes = state_data['unvisited_nodes']

    # If the current solution is empty, start from first unvisited node.
    if not current_solution.tour:
        return AppendOperator(unvisited_nodes[0]), {}

    # If there are no unvisited nodes, return an empty operator
    if not unvisited_nodes:
        return None, {}

    # Determine the node to insert and its position based on the chosen strategy
    node_to_insert = None
    position_to_insert = None
    min_cost_increase = float('inf')

    for node in unvisited_nodes:
        for i in range(len(current_solution.tour) + 1):
            # Calculate the cost increase for inserting the node at position i
            if i == 0:
                prev_node = current_solution.tour[-1]
                next_node = current_solution.tour[0]
            else:
                prev_node = current_solution.tour[i - 1]
                next_node = current_solution.tour[i % len(current_solution.tour)]

            cost_increase = (distance_matrix[prev_node][node] +
                             distance_matrix[node][next_node] -
                             distance_matrix[prev_node][next_node])

            # Update the node and position to insert based on the strategy
            if insertion_strategy == 'cheapest' and cost_increase < min_cost_increase:
                node_to_insert = node
                position_to_insert = i
                min_cost_increase = cost_increase
            elif insertion_strategy == 'farthest' and cost_increase > min_cost_increase:
                node_to_insert = node
                position_to_insert = i
                min_cost_increase = cost_increase
            elif insertion_strategy == 'nearest' and distance_matrix[node][next_node] < min_cost_increase:
                node_to_insert = node
                position_to_insert = i
                min_cost_increase = distance_matrix[node][next_node]

    # Create and return the insert operator with the chosen node and position
    return InsertOperator(node_to_insert, position=position_to_insert), {}