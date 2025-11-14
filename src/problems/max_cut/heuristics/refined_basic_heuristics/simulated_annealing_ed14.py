from src.problems.max_cut.components import *
import random
import math

def simulated_annealing_ed14(problem_state: dict, algorithm_data: dict, initial_temperature: float=100.0, final_temperature: float=0.0, alpha: float=0.95, **kwargs) -> tuple[SwapOperator, dict]:
    """Single-vertex-flip simulated annealing with uniform random neighbor selection and external state evaluation.
    
    At each call, uniformly selects one currently assigned vertex (fallback: any vertex if both sets are empty) and proposes flipping it to the opposite partition via SwapOperator. The proposed neighbor is evaluated using problem_state['get_problem_state'] to obtain its cut value; the weight matrix is not accessed directly. The move is accepted using the Metropolis criterion: accept if delta ≥ 0, else with probability exp(delta / T). Temperature is multiplicatively cooled T ← T * alpha every call, regardless of acceptance. The heuristic returns a SwapOperator only when the move is accepted; otherwise returns None. If temperature is at or below the final threshold, no move is proposed.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "node_num" (int): Total number of vertices; used for fallback selection when both sets are empty.
            - "current_solution" (Solution): Current MaxCut solution; used to construct the neighbor via SwapOperator.
            - "current_cut_value" (int or float): Current cut value; used to compute the delta against the neighbor.
            - "get_problem_state" (callable): def get_problem_state(solution: Solution) -> dict. Returns a state dict that includes at least "current_cut_value" for the provided solution. Used to evaluate the proposed neighbor.
            - "selected_nodes" (set[int]): Not used in this heuristic.
            - "set_a_count" (int): Not used in this heuristic.
            - "set_b_count" (int): Not used in this heuristic.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary / optional:
            - "temperature" (float): Optional. Current temperature; if missing, initialized to initial_temperature.
            - "cooling_rate" (float): Not used in this heuristic (cooling is controlled by alpha parameter).
        Hyperparameters:
            - initial_temperature (float): Starting temperature T0 (> 0). Default is 100.0.
            - final_temperature (float): Stop threshold; when T ≤ final_temperature, no move is proposed. Default is 0.0 (non-negative).
            - alpha (float): Multiplicative cooling factor in (0, 1]; T ← T * alpha each call. Default is 0.95.

    Returns:
        SwapOperator: The operator to swap a single node between sets when the move is accepted; otherwise None if rejected or annealing has halted.
        dict: Updated algorithm data containing:
            - "temperature" (float): Temperature after cooling.
            - "accepted" (bool): Whether the proposed move was accepted.
            - "delta" (float): The change in cut value (neighbor - current).
            - "last_node" (int): The node index proposed for flipping.
    """
    # Initialize temperature from algorithm_data or hyperparameter
    temperature = algorithm_data.get('temperature', initial_temperature)

    # Halt if annealing finished
    if temperature <= final_temperature:
        return None, {'temperature': temperature, 'accepted': False, 'delta': 0.0, 'last_node': None}

    current_solution = problem_state['current_solution']
    current_cut_value = problem_state['current_cut_value']

    # Prefer selecting from assigned vertices to avoid no-op; fallback to any vertex if both sets empty
    assigned_nodes = list(current_solution.set_a | current_solution.set_b)
    if assigned_nodes:
        node = random.choice(assigned_nodes)
    else:
        node = random.randint(0, problem_state['node_num'] - 1)

    # Propose neighbor by flipping the chosen node
    proposed_operator = SwapOperator([node])
    new_solution = proposed_operator.run(current_solution)

    # Evaluate neighbor using external problem state getter
    new_problem_state = problem_state['get_problem_state'](new_solution)
    if new_problem_state is None or 'current_cut_value' not in new_problem_state:
        # If evaluation fails or lacks required metric, cool and return no move
        new_temperature = temperature * alpha
        return None, {'temperature': new_temperature, 'accepted': False, 'delta': 0.0, 'last_node': node}

    new_cut_value = new_problem_state['current_cut_value']
    delta = new_cut_value - current_cut_value

    # Metropolis acceptance
    accepted = False
    if delta >= 0:
        accepted = True
    else:
        # Avoid division-by-zero because we already guard temperature > final_temperature
        acceptance_probability = math.exp(delta / max(temperature, 1e-12))
        if random.random() < acceptance_probability:
            accepted = True

    # Cool temperature regardless of acceptance
    new_temperature = temperature * alpha

    if accepted:
        return proposed_operator, {'temperature': new_temperature, 'accepted': True, 'delta': float(delta), 'last_node': node}
    else:
        return None, {'temperature': new_temperature, 'accepted': False, 'delta': float(delta), 'last_node': node}