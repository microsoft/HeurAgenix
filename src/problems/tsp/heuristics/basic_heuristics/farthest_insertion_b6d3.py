from src.problems.tsp.components import *

def farthest_insertion_b6d3(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertOperator, dict]:
    """ 
    Farthest-insertion with max–max selection on directed distances. At each step, among unvisited nodes, it selects the node whose maximum distance to any node in the current tour is largest (contrast with the classical max–min rule that uses distance to the nearest tour node). The selected node is inserted into the existing cycle at the position minimizing the marginal cost d[i,node] + d[node,j] − d[i,j] over consecutive tour edges (i,j), treating the current tour as a closed cycle throughout construction. Directed costs are used in both selection and insertion, making it suitable for asymmetric TSP. Deterministic seeding with the first unvisited node. Time per step: O(|unvisited|·|tour| + |tour|); constant extra memory. Bias: aggressively favors nodes that are extreme relative to any tour node, promoting rapid expansion toward outliers.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): The distance matrix between nodes.
            - "current_solution" (Solution): The current solution of the TSP.
            - "unvisited_nodes" (list[int]): The list of nodes that have not been visited.
            
        algorithm_data (dict): Contains any data specific to how the algorithm should function.
            - This heuristic does not use algorithm_data.
    
    Returns:
        InsertOperator: The operator to insert the farthest node into the current solution.
        dict: Empty dictionary as this heuristic does not update algorithm_data.
    """
    
    # Extract necessary data from problem_state
    distance_matrix = problem_state["distance_matrix"]
    current_solution = problem_state["current_solution"]
    unvisited_nodes = problem_state["unvisited_nodes"]
    
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