from src.problems.max_cut.components import Solution, InsertNodeOperator, SwapOperator
import numpy as np

def balanced_cut_c0e6(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[InsertNodeOperator, dict]:
    """Balanced Cut heuristic for the Max Cut problem with improvements to ensure balanced partitions, 
    future impact consideration, and periodic swap operations.

    Args:
        global_data (dict): The global data dict containing the following items:
            - weight_matrix (numpy.ndarray): A 2D array representing the weight between nodes.
        state_data (dict): The state dictionary containing the current state information:
            - current_solution (Solution): The current partition of the graph into sets A and B.
            - unselected_nodes (set[int]): The set of nodes that have not yet been selected.
            - set_a (set[int]): The set of nodes in partition A.
            - set_b (set[int]): The set of nodes in partition B.
            - current_cut_value (float): The current cut value of the solution.
        algorithm_data (dict): The algorithm dictionary for the current algorithm only. In this algorithm, the following items are necessary:
            - operation_count (int): The number of operations performed so far. Default is 0.
            - sorted_nodes (list of tuples): A sorted list of (node, future_impact) in descending order. Default is empty.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution. It does not modify the original solution.
        kwargs (dict): Hyperparameters for the heuristic:
            - scaling_factor (float, optional): A hyperparameter to scale the future impact of nodes. Defaults to 0.5.
            - swap_frequency (int, optional): Frequency (in terms of operations) at which swap operations are considered. Defaults to 5.

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

    # Hyperparameters
    scaling_factor = kwargs.get("scaling_factor", 0.5)
    swap_frequency = kwargs.get("swap_frequency", 5)

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
        if best_pair:
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

    # Choose the set that gives the maximum adjusted increase in cut value
    target_set = "A" if adjusted_increase_a >= adjusted_increase_b else "B"

    # Create the operator to insert the selected node into the chosen set
    operator = InsertNodeOperator(best_node, target_set)

    # Return the operator and the updated algorithm data
    return operator, {"sorted_nodes": sorted_nodes, "operation_count": operation_count + 1}