from src.problems.max_cut.components import *
import random
import math

def softmax_gain_insertion_76de(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertNodeOperator, dict]:
    """Stochastic constructive insertion with softmax over per-node, per-side gains and multiplicative cooling.
    
    Core idea:
    - For each unselected node u, compute two immediate cut-gain candidates:
        ΔA(u) = sum_{v in B} w(u,v)  (placing u into set A contributes edges to B)
        ΔB(u) = sum_{v in A} w(u,v)  (placing u into set B contributes edges to A)
      Because the graph is undirected, w(u,v) == w(v,u) and gains are symmetric by construction.
    - Build a sampling distribution over all (node, side) pairs with probabilities proportional to exp(gain / T),
      where T is a temperature parameter controlling exploration vs. exploitation. Lower T biases toward the highest-gain
      action; higher T approaches uniform sampling.
    - Draw one (node, side) according to this distribution and return an InsertNodeOperator placing the node on the sampled side.
    - Apply multiplicative cooling T ← T * alpha and return the updated temperature via the algorithm_data output.
    
    Uniqueness/notes:
    - Uses a numerically stable softmax by subtracting the maximum gain (log-sum-exp trick) during exponentiation.
    - Includes a "greedy when cold" safeguard: if T is below a small threshold, the algorithm deterministically picks the
      global argmax (node, side) instead of sampling, avoiding numerical underflow and ensuring fast exploitation.
    - Validity is guaranteed by only selecting from unselected_nodes and by using InsertNodeOperator which enforces
      disjointness of sets A and B.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): Symmetric adjacency/weight matrix for an undirected graph; weight_matrix[u, v] = weight of edge (u, v).
            - "current_solution" (Solution): Current partition (set_a, set_b). May be empty; sums over empty sets yield zero gains.
            - "unselected_nodes" (set[int]): Nodes that are not yet placed in either set. Sampling is performed over only these nodes.
        algorithm_data (dict): The algorithm dictionary for this heuristic. In this algorithm, the following items are optionally used:
            - "temperature" (float): Current temperature T. If absent, it is initialized from the kwargs default.
        Hyper-parameters in kwargs (all have defaults and may be omitted):
            - "initial_temperature" (float, default=1.0): Starting temperature used if algorithm_data does not provide one.
            - "min_temperature" (float, default=1e-3): Threshold below which the heuristic switches to greedy argmax selection.
            - "alpha" (float, default=0.95): Multiplicative cooling factor applied after each call (T ← T * alpha).
            - "greedy_when_cold" (bool, default=True): If True and T <= min_temperature, pick deterministic best (node, side) instead of sampling.

    Returns:
        InsertNodeOperator: Operator inserting one unselected node into the sampled target set (A or B). Ensures feasibility (no re-insertion).
        dict: Updated algorithm data containing the cooled temperature {'temperature': new_T}. If no move is possible, returns {}.

    Workflow and edge cases:
    - If unselected_nodes is empty, returns (None, {}) because no insertion is possible.
    - Gains are computed as plain Python sums over the respective opposite set; empty sets yield gain 0, which is well-defined.
    - Softmax sampling uses a stable formulation to prevent overflow/underflow; when extremely cold (T <= min_temperature),
      the algorithm selects the maximum-gain action deterministically.
    - The operator is always valid: it never inserts a node already belonging to the opposite set, and only picks from unselected_nodes.
    """

    # Extract necessary data without modifying problem_state
    weight_matrix = problem_state.get("weight_matrix", None)
    current_solution = problem_state.get("current_solution", None)
    unselected_nodes = problem_state.get("unselected_nodes", set())

    # If there are no unselected nodes, no constructive insertion is possible.
    if not unselected_nodes:
        return None, {}

    # Hyper-parameters with safe defaults
    initial_temperature = kwargs.get("initial_temperature", 1.0)  # Exploration-friendly but not too large
    min_temperature = kwargs.get("min_temperature", 1e-3)         # Cold threshold to switch to greedy argmax
    alpha = kwargs.get("alpha", 0.95)                             # Cooling multiplier
    greedy_when_cold = kwargs.get("greedy_when_cold", True)       # Greedy step when temperature is too low

    # Initialize/read current temperature from algorithm_data
    T = algorithm_data.get("temperature", initial_temperature)

    # Access current sets; they may be empty (valid partial solution)
    set_a = current_solution.set_a
    set_b = current_solution.set_b

    # Build candidate list of (node, target_set, gain)
    # Gains:
    # - If we insert node into A, it contributes edges crossing to B: gain_A = sum_{v in B} w(node, v)
    # - If we insert node into B, it contributes edges crossing to A: gain_B = sum_{v in A} w(node, v)
    candidates = []
    for node in unselected_nodes:
        # Sum over set_b for gain to A; handles empty sets by yielding 0
        gain_A = 0.0
        if set_b:
            # Undirected graph: use w(node, v); symmetric matrix expected but we rely only on given values
            gain_A = sum(float(weight_matrix[node, v]) for v in set_b)

        # Sum over set_a for gain to B; handles empty sets by yielding 0
        gain_B = 0.0
        if set_a:
            gain_B = sum(float(weight_matrix[node, v]) for v in set_a)

        candidates.append((node, 'A', gain_A))
        candidates.append((node, 'B', gain_B))

    # If temperature is at or below the minimum and greedy_when_cold is enabled, pick the global argmax deterministically.
    if greedy_when_cold and T <= min_temperature:
        # Select the (node, side) with the maximum gain; ties broken by encounter order.
        best_node, best_side, best_gain = max(candidates, key=lambda x: x[2])
        # Construct a valid operator: node is guaranteed unselected, and the operator enforces disjointness
        op = InsertNodeOperator(node=best_node, target_set=best_side)
        # Cool temperature for the next call
        new_T = T * alpha
        return op, {'temperature': new_T}

    # Otherwise, perform softmax sampling with numerical stability.
    # Stabilize exponentials by subtracting the maximum gain from all gains: exp((g - g_max) / T)
    # If T is non-positive (e.g., due to misconfigured inputs), clamp locally to min_temperature to avoid division by zero.
    if T <= 0.0:
        T = min_temperature

    max_gain = max(c[2] for c in candidates)
    # Compute unnormalized weights
    weights = [math.exp((c[2] - max_gain) / T) for c in candidates]
    total_weight = sum(weights)

    # As there is at least one candidate, total_weight > 0 (since at least exp(0) for max gain).
    # Sample according to these weights using a cumulative distribution.
    r = random.random() * total_weight
    cumulative = 0.0
    chosen_idx = 0
    for idx, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            chosen_idx = idx
            break

    chosen_node, chosen_side, _ = candidates[chosen_idx]
    op = InsertNodeOperator(node=chosen_node, target_set=chosen_side)

    # Multiplicative cooling for next invocation
    new_T = T * alpha
    return op, {'temperature': new_T}