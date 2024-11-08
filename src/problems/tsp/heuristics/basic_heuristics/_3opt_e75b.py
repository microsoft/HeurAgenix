def _3opt_e75b(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ReverseSegmentOperator, dict]:
    """ The 3-opt heuristic operates by breaking the tour into three segments and then attempting to reconnect these segments in a different order that reduces the total travel distance. By considering various reconnection possibilities and implementing the most beneficial rearrangement, the heuristic can significantly reduce the length of the tour. This process is repeated until no further improvements can be found, resulting in a locally optimized solution that is often shorter than the initial tour provided to the function.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.

    Returns:
        ReverseSegmentOperator: The operator that defines the 3-opt move to be performed on the current solution.
        dict: Empty dictionary as no algorithm data is updated.
    """
    distance_matrix = global_data["distance_matrix"]  
    current_solution = state_data["current_solution"]  
  
    # Best distance improvement found  
    best_delta = 0
    best_segments = None  
  
    n = len(current_solution.tour)  
  
    # Convert tour to nodes and precompute a list of node indices  
    nodes = current_solution.tour
  
    # Iterate over all possible combinations of three edges in the tour  
    for i in range(n):  
        for j in range(i + 2, n):  
            for k in range(j + 2, n + (i > 0)):  # Ensure we don't create a sub-loop  
                if k + 1 >= n:
                    continue
                # Calculate the cost difference if these three edges are removed and reconnected  
                a, b = nodes[i], nodes[(i + 1) % n]  
                c, d = nodes[j], nodes[(j + 1) % n]  
                e, f = nodes[k % n], nodes[(k + 1) % n]  
  
                # Calculate the cost of the existing edges  
                current_cost = distance_matrix[a][b] + distance_matrix[c][d] + distance_matrix[e][f]  
  
                # Calculate the cost of the new edges for each option  
                option_1_cost = distance_matrix[a][d] + distance_matrix[c][f] + distance_matrix[e][b]  
                option_2_cost = distance_matrix[a][e] + distance_matrix[d][b] + distance_matrix[c][f]  
                option_3_cost = distance_matrix[a][d] + distance_matrix[e][b] + distance_matrix[c][f] 

  
                # Determine which of the three options is the best  
                best_option_cost = min(option_1_cost, option_2_cost, option_3_cost)  
                delta = best_option_cost - current_cost  
  
                # If this is the best improvement so far, remember it  
                if delta < best_delta:  
                    best_delta = delta  
                    if best_option_cost == option_1_cost:  
                        best_segments = [((i + 1) % n, (j + 1) % n), ((j + 1) % n, (k + 1) % n)]  
                    elif best_option_cost == option_2_cost:  
                        best_segments = [((i + 1) % n, j), ((i + 1) % n, k % n)]  
                    elif best_option_cost == option_3_cost:   
                        best_segments = [((i + 1) % n, (j + 1) % n), (j, k % n)]
  
    # If a move was found that improves the tour, create the corresponding ReverseSegmentOperator  
    if best_segments:  
        # Define the ReverseSegmentOperator logic for the 3-opt move  
        return ReverseSegmentOperator(segments=best_segments), {}  
    else:  
        # No improvement found, return an empty operator  
        return None, {}  