from src.problems.max_cut.components import Solution, InsertNodeOperator

def most_weight_neighbors_320c(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertNodeOperator, dict]:
    """
    Weighted-degree ordered constructive insertion with per-node best side placement. Ranks all unselected nodes once by total incident weight (row-sum/strength) and caches this static ordering in algorithm_data (sorted_nodes). At each step, pops the current top-strength node and places it on the side that maximizes immediate cut gain: gain(A)=sum of weights to current set B, gain(B)=sum of weights to current set A; ties go to A. This is not global best-improvement over all nodes w.r.t. the current partition (the ranking ignores evolving marginal gains), but a low-overhead greedy guided by node strength, suitable for fast construction. Incremental state reuse via filtered sorted_nodes avoids re-sorting across calls. Per call: O(|A|+|B|) to evaluate side gains; initial ranking: O(|U|·n). Constant extra memory beyond the cached node list.
    
    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): A 2D array representing the weight between nodes.
            - "current_solution" (Solution): The current partition of the graph.
            - "unselected_nodes" (set[int]): The set of unselected nodes.
        algorithm_data (dict): Contains the algorithm-specific data. For this algorithm, we use:
            - "sorted_nodes" (list of tuples): A sorted list of (node, weight_sum) in descending order.
        problem_state["get_problem_state"] (callable): Function to get state data for a new solution; not used directly in this algorithm.
        
    Returns:
        (InsertNodeOperator): Operator to insert the selected node into one of the sets.
        (dict): Updated algorithm data with the sorted list of nodes.
    """
    
    weight_matrix = problem_state["weight_matrix"]
    current_solution = problem_state["current_solution"]
    unselected_nodes = problem_state["unselected_nodes"]
    
    # Check if we already have a sorted list of nodes in algorithm_data
    if "sorted_nodes" not in algorithm_data or not algorithm_data["sorted_nodes"]:
        # Sort the unselected nodes based on their connected weights sum in descending order
        sorted_nodes = sorted(
            [(node, sum(weight_matrix[node])) for node in unselected_nodes],
            key=lambda x: x[1],
            reverse=True
        )
        algorithm_data["sorted_nodes"] = sorted_nodes
    else:
        # Filter out the nodes that have been selected since the last run
        sorted_nodes = [
            (node, weight_sum) for (node, weight_sum) in algorithm_data["sorted_nodes"]
            if node in unselected_nodes
        ]

    # Edge case: if there are no unselected nodes, return None.
    if not sorted_nodes:
        return None, {}

    # Select the best node from the sorted list
    best_node, _ = sorted_nodes.pop(0)
    
    # Calculate the potential increase in cut value for adding the node to each set
    potential_increase_a = sum(weight_matrix[best_node][other] for other in current_solution.set_b)
    potential_increase_b = sum(weight_matrix[best_node][other] for other in current_solution.set_a)
    
    # Choose the set that gives the maximum increase in cut value
    target_set = "A" if potential_increase_a >= potential_increase_b else "B"

    # Create and return the operator to insert the selected node into the chosen set.
    return InsertNodeOperator(best_node, target_set), {"sorted_nodes": sorted_nodes}