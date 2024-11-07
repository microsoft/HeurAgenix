from src.problems.tsp.components import *

def cheapest_insertion_605f(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """
    This heuristic selects the non-tour city that, when inserted into the current tour, results in the smallest possible increase in the total tour cost. It then returns an operator that performs this insertion.

    Args:
        global_data (dict): Contains the global data necessary for the heuristic.
            - "distance_matrix" (numpy.ndarray): 2D array with distances between nodes.
            
        state_data (dict): Contains the current state information.
            - "current_solution" (Solution): The current tour solution.
            - "unvisited_nodes" (list[int]): List of node IDs that have not been visited.
            
        algorithm_data (dict): Contains data specific to the algorithm. Not used in this heuristic.

    Returns:
        InsertOperator: The operator to insert the cheapest node into the current solution.
        dict: Empty dictionary as this heuristic does not update algorithm_data.
    """
    distance_matrix = global_data['distance_matrix']
    current_solution = state_data['current_solution']
    unvisited_nodes = state_data['unvisited_nodes']

    # If the current solution is empty, start from first unvisited node.
    if not current_solution.tour:
        return AppendOperator(unvisited_nodes[0]), {}

    # If there are no unvisited nodes, return an empty operator
    if not unvisited_nodes:
        return None, {}

    # Initialize variables to track the cheapest insertion
    cheapest_cost_increase = float('inf')
    cheapest_node = None
    cheapest_position = None

    # Iterate over each unvisited node to find the cheapest insertion
    for node in unvisited_nodes:
        for i in range(len(current_solution.tour) + 1):
            # Calculate the cost increase for inserting the node at position i
            if i == 0:  # Inserting at the beginning
                next_node = current_solution.tour[0]
                cost_increase = distance_matrix[node][next_node]
            elif i == len(current_solution.tour):  # Inserting at the end
                prev_node = current_solution.tour[-1]
                cost_increase = distance_matrix[prev_node][node]
            else:  # Inserting in the middle
                prev_node = current_solution.tour[i - 1]
                next_node = current_solution.tour[i]
                cost_increase = (distance_matrix[prev_node][node] +
                                 distance_matrix[node][next_node] -
                                 distance_matrix[prev_node][next_node])

            # Update the cheapest insertion if this is the lowest cost increase found so far
            if cost_increase < cheapest_cost_increase:
                cheapest_cost_increase = cost_increase
                cheapest_node = node
                cheapest_position = i

    # Create the operator to perform the cheapest insertion
    insert_operator = InsertOperator(node=cheapest_node, position=cheapest_position)

    return insert_operator, {}