from src.problems.max_cut.components import Solution, InsertNodeOperator
import random

def random_5c59(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertNodeOperator, dict]:
    """
    Stochastic diversification move for partial MaxCut constructions. Uniformly samples one unassigned vertex and assigns it to a random side (A or B) with equal probability, without evaluating cut-gain or partition balance. Pure exploration: ignores edge weights and current cut value, providing unbiased diversification irrespective of current set sizes. Feasibility preserved by design (vertex belongs to exactly one set). Suitable as a warm-start builder, perturbation step in ILS/VNS, or population diversification in metaheuristics. No improvement guarantee; effectiveness arises when followed by gain-based local search (e.g., single-vertex flip). Time complexity per call: O(1) aside from sampling; O(1) extra memory. Determinism controllable via RNG seeding.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "node_num" (int): The total number of vertices in the graph.
            - "current_solution" (Solution): The current solution instance.
            - "unselected_count" (int): The number of unselected nodes.
    
    Returns:
        InsertNodeOperator: The operator to insert a node into set A or B.
        dict: Empty dictionary as no algorithm data is updated.
    """
    node_num = problem_state['node_num']
    current_solution = problem_state['current_solution']
    unselected_nodes = problem_state['unselected_nodes']

    # If there are no unselected nodes left, return None.
    if not unselected_nodes:
        return None, {}

    # Randomly choose an unselected node.
    node_to_insert = random.choice(list(unselected_nodes))
    
    # Randomly decide to which set the node will be inserted.
    target_set = random.choice(['A', 'B'])

    # Create the operator.
    operator = InsertNodeOperator(node=node_to_insert, target_set=target_set)
    
    # Return the operator and an empty algorithm data dictionary.
    return operator, {}