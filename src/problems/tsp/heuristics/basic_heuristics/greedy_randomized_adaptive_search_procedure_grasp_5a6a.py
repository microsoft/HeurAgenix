import random
import numpy as np
from src.problems.tsp.components import *

def greedy_randomized_adaptive_search_procedure_grasp_5a6a(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, alpha: float = 0.3, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Greedy Randomized Adaptive Search Procedure (GRASP) function embodies the Greedy Randomized Adaptive Search Procedure (GRASP), a metaheuristic algorithm tailored for solving combinatorial optimization problems like the Traveling Salesman Problem (TSP). GRASP iteratively constructs a solution through a two-phase process: a greedy randomized construction phase and a local search phase. During the construction phase, it builds a partial solution by repeatedly adding the best candidate according to a greedy function, which is randomized within a Restricted Candidate List (RCL) determined by the alpha parameter. This parameter adjusts the level of greediness versus randomness, with lower values favoring greedy choices and higher values allowing for more exploration. Once an initial solution is constructed, the local search phase seeks to improve it by exploring the neighborhood of the current solution. 

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - distance_matrix (numpy.ndarray): A 2D array representing the distances between nodes.
            - node_num (int): The total number of nodes in the problem.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): An instance of the Solution class representing the current solution.
            - unvisited_nodes (list[int]): A list of integers representing the IDs of nodes that have not yet been visited.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. This algorithm does not use algorithm_data.
        alpha (float): The greediness-randomness parameter. Default is 0.3. It determines the balance between greedy and random selection of the next node to insert.

    Returns:
        InsertOperator: The operator to insert the next node into the current solution.
        dict: Empty dictionary as this algorithm does not update algorithm_data.
    """

    current_solution = state_data['current_solution']
    unvisited_nodes = state_data['unvisited_nodes']
    distance_matrix = global_data['distance_matrix']

    # If the current solution is empty, start from first unvisited node.
    if not current_solution.tour:
        return AppendOperator(unvisited_nodes[0]), {}
    
    # If there are no unvisited nodes, return an empty operator
    if not unvisited_nodes:
        return None, {}

    # Find the last node in the current solution
    last_node = current_solution.tour[-1] if current_solution.tour else None

    # Calculate the cost to each unvisited node from the last node in the solution
    costs = []
    for unvisited_node in unvisited_nodes:
        if last_node is None:
            # If there is no last node, use the first unvisited node
            cost = 0
        else:
            cost = distance_matrix[last_node][unvisited_node]
        costs.append((unvisited_node, cost))

    # Sort the unvisited nodes by cost (ascending)
    costs.sort(key=lambda x: x[1])

    # Determine the candidate list size based on the alpha parameter
    candidate_list_size = max(1, int(alpha * len(unvisited_nodes)))

    # Select a candidate from the candidate list at random
    candidate = costs[:candidate_list_size]
    selected_node = candidate[np.random.randint(0, len(candidate))][0]

    # Find the best position to insert the selected node
    best_increase = float('inf')
    best_position = 0
    for i in range(len(current_solution.tour) + 1):
        # Calculate the increase in cost if we insert the selected node at position i
        if i == 0:
            increase = distance_matrix[selected_node][current_solution.tour[i]]
        elif i == len(current_solution.tour):
            increase = distance_matrix[current_solution.tour[-1]][selected_node]
        else:
            increase = (distance_matrix[current_solution.tour[i-1]][selected_node] +
                        distance_matrix[selected_node][current_solution.tour[i]] -
                        distance_matrix[current_solution.tour[i-1]][current_solution.tour[i]])
        if increase < best_increase:
            best_increase = increase
            best_position = i

    # Create the insert operator with the selected node and best position
    insert_operator = InsertOperator(selected_node, best_position)

    return insert_operator, {}