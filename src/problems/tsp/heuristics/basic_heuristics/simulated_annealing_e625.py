from src.problems.tsp.components import *
import random
import math

def simulated_annealing_e625(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[SwapOperator, dict]:
    """
    Stochastic node-interchange simulated annealing on a closed cyclic tour. Each call samples a uniformly random pair of distinct positions and proposes swapping the corresponding node IDs (not a 2-opt edge reversal). The marginal cost is computed by re-evaluating only the four incident edges around each selected position, with cyclic predecessors/successors via modulo indexing; adjacent and wrap-around cases are implicitly included. Distances are queried directionally, making it applicable to asymmetric matrices.
    Acceptance follows the Metropolis criterion: accept improved moves unconditionally and worsenings with probability exp(-Δ/T). Temperature T and cooling factor α are taken from algorithm_data and updated as T ← α·T on every call, independent of acceptance. If accepted, returns a SwapOperator with the chosen node pair; otherwise returns no operator. Per-step complexity is O(1) with constant memory. Requires: distance_matrix, current_solution, current_cost; algorithm_data keys: temperature, alpha (defaults supported).

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
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
    node_indices = list(range(len(problem_state['current_solution'].tour)))
    if len(node_indices) < 2:
        return None, {}
    i, j = random.sample(node_indices, 2)
    
    # Calculate the cost difference if the nodes were swapped
    current_solution = problem_state['current_solution']
    distance_matrix = problem_state['distance_matrix']
    current_cost = problem_state['current_cost']
    
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