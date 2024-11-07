from src.problems.tsp.components import *
import numpy as np

def ant_colony_d4f7(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Ant Colony Optimization heuristic for the Traveling Salesman Problem. This function simulates the behavior of ants laying down pheromones and choosing paths probabilistically to construct a solution for the TSP.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - distance_matrix (numpy.ndarray): A 2D array representing the distances between nodes.
            - node_num (int): The total number of nodes in the problem.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): An instance of the Solution class representing the current solution.
            - unvisited_nodes (list[int]): A list of integers representing the IDs of nodes that have not yet been visited.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary:
            - pheromone_levels (numpy.ndarray): A 2D array representing the pheromone levels on the edges between nodes.
            - desirability (numpy.ndarray): A 2D array representing the desirability (heuristic information) of the edges between nodes.

    Returns:
        InsertOperator: The operator to insert the next node into the current solution based on the ant colony optimization.
        dict: Updated algorithm data with the new pheromone levels.
    """
    # Hyperparameters with default values
    alpha = kwargs.get('alpha', 1.0)  # Pheromone importance
    beta = kwargs.get('beta', 2.0)    # Desirability importance
    evaporation_rate = kwargs.get('evaporation_rate', 0.5)
    pheromone_deposit = kwargs.get('pheromone_deposit', 1.0)

    # Extract necessary data from global_data
    distance_matrix = global_data['distance_matrix']
    node_num = global_data['node_num']

    # Extract necessary data from state_data
    current_solution = state_data['current_solution']
    unvisited_nodes = state_data['unvisited_nodes']

    # Extract necessary data from algorithm_data
    pheromone_levels = algorithm_data.get('pheromone_levels', np.ones((node_num, node_num)))
    desirability = 1 / (distance_matrix + 1e-10)  # To avoid division by zero

    # If there are no unvisited nodes, return no operator
    if not unvisited_nodes:
        return None, {}

    # Select the next node based on pheromone levels and desirability
    last_node = current_solution.tour[-1] if current_solution.tour else 0
    probabilities = []
    for next_node in unvisited_nodes:
        pheromone = pheromone_levels[last_node][next_node] ** alpha
        desirability_score = desirability[last_node][next_node] ** beta
        probabilities.append(pheromone * desirability_score)

    # Normalize probabilities
    probabilities = probabilities / np.sum(probabilities)

    # Choose the next node probabilistically
    next_node = np.random.choice(unvisited_nodes, p=probabilities)

    # Create an InsertOperator to insert the chosen node into the solution
    insert_position = len(current_solution.tour)  # Insert at the end of the current solution
    insert_operator = InsertOperator(next_node, insert_position)

    # Update pheromone levels (evaporation and deposit)
    pheromone_levels *= (1 - evaporation_rate)  # Evaporation
    pheromone_levels[last_node][next_node] += pheromone_deposit  # Deposit

    # Return the operator and the updated pheromone levels
    return insert_operator, {'pheromone_levels': pheromone_levels}