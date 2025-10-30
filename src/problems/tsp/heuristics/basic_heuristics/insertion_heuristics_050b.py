from src.problems.tsp.components import *

def insertion_heuristics_050b(problem_state: dict, algorithm_data: dict, insertion_strategy: str = 'cheapest', **kwargs) -> tuple[InsertOperator, dict]:
    """ 
    Unified insertion heuristic evaluated on a cyclic tour. For each unvisited node and each position i in 0..L (L=len(tour)), it defines (prev,next) as (tour[i-1], tour[i % L]) and scores the candidate as a replacement of edge (prev,next), thus always treating endpoints as the wrap-around (last, first). Positions i=0 and i=L are duplicates of the same cut; this yields L+1 enumerations but only L distinct insertion slots, with equivalent cycle but different linearizations.
    Strategy semantics:
        - cheapest: selects the node-position pair minimizing Δ = d(prev,node) + d(node,next) − d(prev,next).
        - farthest: intended to maximize the same Δ, but the comparison uses “>” against an initial +∞ threshold, so no candidate ever updates; as written this branch is non-operative.
        - nearest: uses d(node,next) only as the score, ignoring Δ; this deviates from textbook nearest-insertion (which normally inserts the nearest node then places it by minimal Δ), and biases toward the smallest forward arc to next.
    Constructive-phase oriented yet closure-aware (evaluates the tour as a cycle at all times); agnostic to symmetry (works for ATSP and STSP). Seed: if the tour is empty, append the first unvisited node. Time complexity O(|unvisited|·|tour|); constant extra memory. Algorithm data is unused.


    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - distance_matrix (numpy.ndarray): A 2D array representing the distances between nodes.
            - current_solution (Solution): An instance of the Solution class representing the current solution.
            - unvisited_nodes (list[int]): A list of integers representing the IDs of nodes that have not yet been visited.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. This algorithm does not use algorithm_data.
        insertion_strategy (str): The strategy to use for insertion. Defaults to 'cheapest'. Other valid values are 'farthest' and 'nearest'.

    Returns:
        InsertOperator: The operator to modify the current solution.
        dict: Empty dictionary as this algorithm does not update algorithm_data.
    """
    # Extract necessary data from problem_state
    distance_matrix = problem_state['distance_matrix']
    current_solution = problem_state['current_solution']
    unvisited_nodes = problem_state['unvisited_nodes']

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