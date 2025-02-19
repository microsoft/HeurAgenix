from src.problems.max_cut.components import Solution, InsertNodeOperator, SwapOperator
import numpy as np

def most_weight_neighbors_0919(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, swap_frequency: int = 5, base_threshold: float = 0.1, fixed_scaling_factor: float = 0.5, **kwargs) -> tuple[InsertNodeOperator, dict]:
    """ Heuristic algorithm to optimize the MaxCut problem by adjusting the scaling factor dynamically with an adaptive threshold based on edge weight and density.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): A 2D array representing the weight between nodes.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current solution of the MaxCut problem.
            - "unselected_nodes" (set[int]): The set of unselected nodes.
        algorithm_data (dict): The algorithm dictionary for the current algorithm only. In this algorithm, the following items are necessary:
            - "operation_count" (int): The number of operations performed so far. Default is 0.
            - "sorted_nodes" (list of tuples): A sorted list of (node, future_impact) in descending order. Default is empty.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution. It does not modify the original solution.
        kwargs: 
            - swap_frequency (int, optional): Frequency (in terms of operations) at which swap operations are considered. Defaults to 5.
            - base_threshold (float, optional): Base threshold for determining scaling factor. Defaults to 0.1.
            - fixed_scaling_factor (float, optional): The fixed scaling factor used when conditions apply. Defaults to 0.5.

    Returns:
        InsertNodeOperator or SwapOperator: The operator to modify the solution.
        dict: Updated algorithm data with the sorted list of nodes and operation count.
    """
    
    # Extract necessary data
    weight_matrix = global_data["weight_matrix"]
    current_solution = state_data["current_solution"]
    unselected_nodes = state_data["unselected_nodes"]
    set_a, set_b = current_solution.set_a, current_solution.set_b
    operation_count = algorithm_data.get("operation_count", 0)
    sorted_nodes = algorithm_data.get("sorted_nodes", [])

    # Calculate average edge weight and edge density if not provided
    total_edges = np.count_nonzero(weight_matrix) / 2  # Divide by 2 for undirected graph
    possible_edges = len(weight_matrix) * (len(weight_matrix) - 1) / 2
    edge_density = total_edges / possible_edges
    average_edge_weight = np.sum(weight_matrix) / (2 * total_edges) if total_edges > 0 else 0

    # Calculate standard deviation of edge weights
    std_dev = np.std(weight_matrix)

    # Adapt threshold based on average edge weight and edge density
    adaptive_threshold = base_threshold * (1 + edge_density) * abs(average_edge_weight)

    # Determine scaling factor based on adaptive threshold
    if std_dev < adaptive_threshold:
        scaling_factor = fixed_scaling_factor
    else:
        scaling_factor = 1.0 / (1 + std_dev)  # Dynamic adjustment

    # Step 1: Perform a swap operation periodically
    if operation_count % swap_frequency == 0 and set_a and set_b:
        weight_to_a = weight_matrix[:, list(set_a)].sum(axis=1)
        weight_to_b = weight_matrix[:, list(set_b)].sum(axis=1)
        best_increase = float("-inf")
        best_pair = None

        # Evaluate all possible swaps
        for i in set_a:
            for j in set_b:
                delta = weight_to_a[i] - weight_to_a[j] + weight_to_b[j] - weight_to_b[i]
                if weight_matrix[i, j] != 0:  # Adjust for the edge between i and j if it exists
                    delta += 2 * weight_matrix[i, j]
                if delta > best_increase:
                    best_increase = delta
                    best_pair = (i, j)

        # If a beneficial swap is found, return the SwapOperator
        if best_pair and best_increase > 0:
            return SwapOperator(nodes=list(best_pair)), {"operation_count": operation_count + 1}

    # Step 2: Sort unselected nodes based on future impact if not already sorted
    if not sorted_nodes:
        # Calculate the future impact for each unselected node
        sorted_nodes = sorted(
            [(node, sum(abs(weight_matrix[node][other]) for other in range(len(weight_matrix)))) for node in unselected_nodes],
            key=lambda x: x[1],
            reverse=True
        )
    else:
        # Filter out nodes that have been selected since the last run
        sorted_nodes = [
            (node, future_impact) for (node, future_impact) in sorted_nodes
            if node in unselected_nodes
        ]

    # Step 3: Select the best unselected node based on both immediate and future impacts
    if not sorted_nodes:
        return None, {}

    # Extract the best node and its future impact
    best_node, future_impact = sorted_nodes.pop(0)

    # Calculate the potential increase in cut value for adding the node to each set
    potential_increase_a = sum(weight_matrix[best_node][other] for other in set_b)
    potential_increase_b = sum(weight_matrix[best_node][other] for other in set_a)

    # Adjust the potential increases by adding the scaled future impact
    adjusted_increase_a = potential_increase_a + scaling_factor * future_impact
    adjusted_increase_b = potential_increase_b + scaling_factor * future_impact

    # Consider negative weights critically and choose the set that maximizes the cut value
    if sum(weight_matrix[best_node]) < 0:
        target_set = "A" if adjusted_increase_a > adjusted_increase_b else "B"
    else:
        # Choose the set that gives the maximum adjusted increase in cut value
        target_set = "A" if adjusted_increase_a >= adjusted_increase_b else "B"

    # Create the operator to insert the selected node into the chosen set
    operator = InsertNodeOperator(best_node, target_set)

    # Return the operator and the updated algorithm data
    return operator, {"sorted_nodes": sorted_nodes, "operation_count": operation_count + 1}