from src.problems.tsp.components import *
import random
import math

def simulated_annealing_e625(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[SwapOperator, dict]:
    """
    Simulated Annealing heuristic for the Traveling Salesman Problem.
    This function applies the simulated annealing technique to find a solution to the TSP.
    It randomly selects two nodes and proposes to swap them. The swap is accepted based on
    the Metropolis criterion, which depends on the temperature and the change in cost of the solution.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current solution of the TSP.
            - "current_cost" (int): The total cost of the current solution.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary:
            - "temperature" (float): The current temperature for the simulated annealing process.
            - "alpha" (float): The cooling rate of the temperature.
    
    Returns:
        SwapOperator: The operator that swaps two nodes in the solution.
        dict: Updated algorithm data with the new temperature.
    """
    
    # Hyperparameters with default values
    temperature = algorithm_data.get('temperature', 10)
    alpha = algorithm_data.get('alpha', 0.995)
    
    # Select two distinct nodes at random
    node_indices = list(range(len(state_data['current_solution'].tour)))
    if len(node_indices) < 2:
        return None, {}
    i, j = random.sample(node_indices, 2)
    
    # Calculate the cost difference if the nodes were swapped
    current_solution = state_data['current_solution']
    distance_matrix = global_data['distance_matrix']
    current_cost = state_data['current_cost']
    
    node_i = current_solution.tour[i]
    node_j = current_solution.tour[j]
    
    # Calculate the new cost after swapping
    cost_remove = (
        distance_matrix[node_i, current_solution.tour[(i-1) % len(node_indices)]] +
        distance_matrix[node_i, current_solution.tour[(i+1) % len(node_indices)]] +
        distance_matrix[node_j, current_solution.tour[(j-1) % len(node_indices)]] +
        distance_matrix[node_j, current_solution.tour[(j+1) % len(node_indices)]]
    )

    # Calculate the cost of edges to be added
    cost_add = (
        distance_matrix[node_j, current_solution.tour[(i-1) % len(node_indices)]] +
        distance_matrix[node_j, current_solution.tour[(i+1) % len(node_indices)]] +
        distance_matrix[node_i, current_solution.tour[(j-1) % len(node_indices)]] +
        distance_matrix[node_i, current_solution.tour[(j+1) % len(node_indices)]]
    )

    # Calculate the new cost after swapping
    new_cost = current_cost - cost_remove + cost_add

    # Calculate the cost difference
    cost_diff = new_cost - current_cost
    
    # Decide whether to accept the swap
    if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
        # Create the swap operator
        swap_operator = SwapOperator(swap_node_pairs=[(current_solution.tour[i], current_solution.tour[j])])
    else:
        # No swap is made
        swap_operator = None
    
    # Update the temperature
    new_temperature = temperature * alpha
    updated_algorithm_data = {'temperature': new_temperature}
    
    return swap_operator, updated_algorithm_data