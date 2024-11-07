from src.problems.tsp.components import *

def farthest_insertion_b6d3(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """ This heuristic selects the non-tour city that is farthest from any city in the current tour and inserts it where it causes the least increase in the tour cost.

    Args:
        global_data (dict): Contains the global data necessary for the heuristic.
            - "distance_matrix" (numpy.ndarray): The distance matrix between nodes.
            
        state_data (dict): Contains the current state information necessary for the heuristic.
            - "current_solution" (Solution): The current solution of the TSP.
            - "unvisited_nodes" (list[int]): The list of nodes that have not been visited.
            
        algorithm_data (dict): Contains any data specific to how the algorithm should function.
            - This heuristic does not use algorithm_data.
    
    Returns:
        InsertOperator: The operator to insert the farthest node into the current solution.
        dict: Empty dictionary as this heuristic does not update algorithm_data.
    """
    
    # Extract necessary data from global_data and state_data
    distance_matrix = global_data["distance_matrix"]
    current_solution = state_data["current_solution"]
    unvisited_nodes = state_data["unvisited_nodes"]
    
    # Initialize variables to track the farthest node and its insertion cost
    farthest_node = None
    max_distance_to_tour = -1
    min_insertion_cost = float('inf')
    insert_position = -1
    
    # If the current solution is empty, start from first unvisited node.
    if not current_solution.tour:
        return AppendOperator(unvisited_nodes[0]), {}

    # If there are no unvisited nodes, return an empty operator
    if not unvisited_nodes:
        return None, {}

    # Iterate over unvisited nodes to find the farthest node
    for node in unvisited_nodes:
        for tour_node in current_solution.tour:
            distance_to_tour_node = distance_matrix[node][tour_node]
            if distance_to_tour_node > max_distance_to_tour:
                farthest_node = node
                max_distance_to_tour = distance_to_tour_node
    
    # Find the position in the current tour where inserting the farthest node has the least cost
    for i in range(len(current_solution.tour)):
        next_i = (i + 1) % len(current_solution.tour)
        insertion_cost = (distance_matrix[current_solution.tour[i]][farthest_node] +
                          distance_matrix[farthest_node][current_solution.tour[next_i]] -
                          distance_matrix[current_solution.tour[i]][current_solution.tour[next_i]])
        if insertion_cost < min_insertion_cost:
            min_insertion_cost = insertion_cost
            insert_position = next_i

    # Create the insert operator with the farthest node and the best insertion position
    operator = InsertOperator(node=farthest_node, position=insert_position)
    return operator, {}