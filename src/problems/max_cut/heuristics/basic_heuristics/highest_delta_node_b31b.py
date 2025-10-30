from src.problems.max_cut.components import *

def highest_delta_node_b31b(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertNodeOperator, dict]:
    """
    Greedy global best-improvement insertion for partial MaxCut partitions. For each unassigned node, evaluates both placements: ΔA = sum of weights from the node to current set B; ΔB = sum of weights from the node to current set A. Selects the node and target set with the largest Δ, performing an argmax over all (node, target_set) pairs. Strict-improvement updating: a candidate replaces the incumbent only if its Δ exceeds the current best; equal-Δ ties do not update (first-best retention). When ΔA == ΔB and the pair triggers an update (both strictly > current best), the tie breaks to set B. With an empty partition, the first inserted node goes to set B. Not restricted to positive gains; if all Δ are negative, it inserts the least-worsening node, making it suitable for constructive/repair phases rather than strict local improvement. Uses directed outgoing weights weight_matrix[node, other]; for asymmetric matrices it optimizes the node’s outgoing contribution to the cut. Time complexity: O(|unselected| × (|A| + |B|)); O(1) extra memory.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): A 2D array representing the weight between nodes.
            - "current_solution" (Solution): The current partition of the graph into sets A and B.
            - "unselected_count" (int): The number of nodes not yet selected into either set A or B.
            - "unselected_nodes" (set[int]): The set of unselected nodes.
        algorithm_data (dict): Not used in this heuristic.
        problem_state["get_problem_state"] (callable): Function to get state data for a new solution.

    Returns:
        InsertNodeOperator: The operator to insert the node into the appropriate set.
        dict: Empty dictionary as this algorithm doesn't update the algorithm data.
    """

    # Extract necessary information from problem_state
    weight_matrix = problem_state["weight_matrix"]
    current_solution = problem_state["current_solution"]
    unselected_nodes = problem_state["unselected_nodes"]

    # Initialize variables to keep track of the best node and delta
    best_node = None
    best_delta = -float('inf')

    # Iterate over all unselected nodes to find the one with the highest delta
    for node in unselected_nodes:
        delta_a = sum(weight_matrix[node, other] for other in current_solution.set_b)
        delta_b = sum(weight_matrix[node, other] for other in current_solution.set_a)

        # Check if placing the node in set A or B gives a better delta
        if delta_a > best_delta or delta_b > best_delta:
            best_delta = max(delta_a, delta_b)
            best_node = node
            target_set = 'A' if delta_a > delta_b else 'B'

    # If no suitable node is found (all nodes are already selected), return None
    if best_node is None:
        return None, {}

    # Create the operator to insert the best node into the chosen set
    operator = InsertNodeOperator(best_node, target_set)

    # Return the operator and an empty dictionary as no algorithm data is updated
    return operator, {}