from src.problems.max_cut.components import *
import random
import math

def simulated_annealing_ed14(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[SwapOperator, dict]:
    """
    Single-vertex-flip simulated annealing with uniform random neighbor selection. Each iteration proposes flipping exactly one randomly chosen node to the opposite partition (SwapOperator), without scanning or ranking neighbors—neither first-improvement nor best-improvement is attempted. Move quality (delta) is computed via the external get_problem_state for the proposed solution; the weight matrix is not accessed directly. Acceptance uses the Metropolis criterion: accept if delta ≥ 0, else with probability exp(delta/T). Temperature follows multiplicative cooling T ← T·alpha every iteration regardless of acceptance; the search halts when T ≤ final_temperature. Invalid neighbors (as flagged by get_problem_state) are discarded. If a node is in neither set, the flip is effectively a no-op. One neighbor evaluated per call; runtime dominated by the state evaluation.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "total_nodes" (int): The total number of vertices in the graph.
            - "current_solution" (Solution): The current solution of the Max Cut problem.
            - "current_cut_value" (int or float): The total weight of edges between set A and set B in the current solution.
            - get_problem_state (callable): def validation_solution(solution: Solution) -> bool: The function to get the problem state for given solution without modify it.
        algorithm_data (dict): Contains the data specific to the simulated annealing algorithm.
            - "temperature" (float): The current temperature for the simulated annealing process.
            - "cooling_rate" (float): The rate at which the temperature decreases.
        **kwargs: Hyperparameters for the algorithm.
            - "initial_temperature" (float): The starting temperature for the annealing process.
            - "final_temperature" (float): The temperature at which the annealing process stops.
            - "alpha" (float): The cooling rate factor.

    Returns:
        SwapOperator: The operator to swap a node between sets if a valid move is found.
        dict: Updated algorithm data with the new temperature.
    """
    # Hyperparameters with default values
    initial_temperature = kwargs.get('initial_temperature', 100.0)
    final_temperature = kwargs.get('final_temperature', 0.001)
    alpha = kwargs.get('alpha', 0.95)

    # Initialize temperature if not present in algorithm_data
    temperature = algorithm_data.get('temperature', initial_temperature)
    cooling_rate = algorithm_data.get('cooling_rate', alpha)

    # Current solution and cut value
    current_solution = problem_state['current_solution']
    current_cut_value = problem_state['current_cut_value']

    # If the temperature is already below the final temperature, do not perform any operation
    if temperature <= final_temperature:
        return None, {}

    # Select a random node to swap
    node = random.randint(0, problem_state['node_num'] - 1)

    # Calculate delta for single node flip manually (Avoid creating new Solution and full scan)
    adj = problem_state["adj"] # Using adjacency list for fast delta calculation
    current_solution = problem_state["current_solution"]
    set_a = current_solution.set_a
    set_b = current_solution.set_b
    
    delta = 0
    # Logic: if node in A, moving to B means gaining edges to A, losing edges to B
    if node in set_a:
        for neighbor, weight in adj[node].items():
            if neighbor in set_a:
                delta += weight # Gain (now connected to new opposite)
            elif neighbor in set_b:
                delta -= weight # Loss (now connected to same side)
    elif node in set_b:
        for neighbor, weight in adj[node].items():
            if neighbor in set_b:
                delta += weight
            elif neighbor in set_a:
                delta -= weight
    else:
        # Node not assigned yet, cannot flip.
        # But this is SA, usually working on complete solutions.
        return None, {'temperature': temperature * cooling_rate}


    # If the new solution is better or equal, accept it
    if delta >= 0:
        return SwapOperator([node]), {'temperature': temperature * cooling_rate}

    # If the new solution is worse, accept it with a certain probability
    acceptance_probability = math.exp(delta / temperature)
    if random.random() < acceptance_probability:
        return SwapOperator([node]), {'temperature': temperature * cooling_rate}

    # If the new solution is not accepted, return no operation
    return None, {'temperature': temperature * cooling_rate}