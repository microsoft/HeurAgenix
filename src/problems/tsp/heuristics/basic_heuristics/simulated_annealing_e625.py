from src.problems.tsp.components import *
import random
import math

def simulated_annealing_e625(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[SwapOperator, dict]:
    """
    Stochastic node-interchange simulated annealing on a closed cyclic tour. Each call samples a uniformly random pair of distinct positions and proposes swapping the corresponding nodes (position swap, not 2-opt). The marginal cost Δ is computed by re-evaluating only the incident tour edges around the two positions with modulo indexing; adjacent cases are handled as 3-edge changes to avoid double counting. Distances are queried directionally (prev→node, node→next), making it applicable to asymmetric matrices.

    Acceptance follows the Metropolis criterion: accept improved moves (Δ<0) unconditionally and worsenings with probability exp(-Δ/T). Temperature T and cooling factor α are taken from algorithm_data and updated as T ← α·T on every call, independent of acceptance. If accepted, returns a SwapOperator with the chosen node pair; otherwise returns no operator (None). Per-step complexity is O(1) with constant memory. Requires: distance_matrix, current_solution, current_cost; algorithm_data keys: temperature, alpha (defaults calibrated to problem_state if absent).

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
            - "current_solution" (Solution): The current solution of the TSP.
            - "current_cost" (int): The total cost of the current solution.
            - Optional for defaults: "std_dev_distance" or "average_edge_cost", used to scale the initial temperature if missing.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary:
            - "temperature" (float): The current temperature for the simulated annealing process (default scaled from problem_state).
            - "alpha" (float): The cooling rate of the temperature (default 0.995).
    
    Returns:
        SwapOperator | None: The operator that swaps two nodes in the solution if accepted; otherwise None.
        dict: Updated algorithm data with the new temperature (and preserved alpha).
    """

    # Imports (ensure available)
    # import random, math

    # Hyperparameters with calibrated defaults
    temperature = algorithm_data.get('temperature',
                                     problem_state.get('std_dev_distance',
                                                       problem_state.get('average_edge_cost', 10.0)))
    alpha = algorithm_data.get('alpha', 0.995)

    # Select two distinct positions
    n = len(problem_state['current_solution'].tour)
    if n < 2 or temperature <= 0:
        return None, {'temperature': temperature, 'alpha': alpha}
    i, j = random.sample(range(n), 2)

    tour = problem_state['current_solution'].tour
    dm = problem_state['distance_matrix']
    current_cost = problem_state['current_cost']

    # Neighbors and nodes
    i_prev = tour[(i - 1) % n]
    i_node = tour[i]
    i_next = tour[(i + 1) % n]
    j_prev = tour[(j - 1) % n]
    j_node = tour[j]
    j_next = tour[(j + 1) % n]

    # Δcost (directional) with adjacency handling
    if (j == (i + 1) % n):  # j is next of i
        cost_remove = dm[i_prev, i_node] + dm[i_node, j_node] + dm[j_node, j_next]
        cost_add    = dm[i_prev, j_node] + dm[j_node, i_node] + dm[i_node, j_next]
    elif (i == (j + 1) % n):  # i is next of j
        cost_remove = dm[j_prev, j_node] + dm[j_node, i_node] + dm[i_node, i_next]
        cost_add    = dm[j_prev, i_node] + dm[i_node, j_node] + dm[j_node, i_next]
    else:
        cost_remove = dm[i_prev, i_node] + dm[i_node, i_next] + dm[j_prev, j_node] + dm[j_node, j_next]
        cost_add    = dm[i_prev, j_node] + dm[j_node, i_next] + dm[j_prev, i_node] + dm[i_node, j_next]

    cost_diff = (current_cost - cost_remove + cost_add) - current_cost

    # Metropolis acceptance
    if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
        swap_operator = SwapOperator(swap_node_pairs=[(i_node, j_node)])
    else:
        swap_operator = None

    # Cooling and algorithm data update (preserve alpha)
    new_temperature = temperature * alpha
    updated_algorithm_data = dict(algorithm_data)
    updated_algorithm_data['temperature'] = new_temperature
    updated_algorithm_data['alpha'] = alpha

    return swap_operator, updated_algorithm_data