from src.problems.tsp.components import *

def two_opt_89aa(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[ReverseSegmentOperator, dict]:
    """
    Single-move 2-opt with best-improvement on a closed, symmetric tour. Enumerates all non-adjacent edge pairs (i, i+1) and (j, j+1), skipping the wrap pair (0, n−1), and uses modular indexing to treat the tour as a cycle. Evaluates the exact 2-edge exchange delta: Δ = d(a,c) + d(b,d) − [d(a,b) + d(c,d)], tracking the most negative Δ over the full scan (best, not first improvement). If an improving pair is found, applies a single segment reversal on [i+1 .. j], which realizes the exchange. Assumes a symmetric (undirected) cost matrix so that reversing a subpath changes only the two boundary edges; not suitable for asymmetric costs as written. Produces at most one improving move per invocation; repeat until no improving pair exists to reach a 2-opt local optimum. Time complexity per invocation: O(n^2); O(1) extra memory.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "current_cost" (int): The total cost of current solution.

    Returns:
        ReverseSegmentOperator: The operator that reverse two nodes in the solution to achieve a shorter tour.
        dict: Empty dictionary as this algorithm does not update algorithm_data.
    """
    distance_matrix = problem_state["distance_matrix"]
    current_solution = problem_state["current_solution"]
    current_cost = problem_state["current_cost"]

    # Best improvement setup
    best_delta = 0
    best_pair = None

    # Iterate over all pairs of indices to consider removing
    for i in range(len(current_solution.tour) - 1):  
            for j in range(i + 2, len(current_solution.tour)):  
                if j == len(current_solution.tour) - 1 and i == 0:  
                    continue

                # Calculate the cost difference if these two edges are removed and reconnected
                a, b = current_solution.tour[i], current_solution.tour[(i + 1) % len(current_solution.tour)]
                c, d = current_solution.tour[j], current_solution.tour[(j + 1) % len(current_solution.tour)]
                current_cost = distance_matrix[a][b] + distance_matrix[c][d]
                new_cost = distance_matrix[a][c] + distance_matrix[b][d]
                delta = new_cost - current_cost

                # Check for an improvement
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (i + 1, j)

    # If an improvement has been found, create and return the corresponding SwapOperator
    if best_pair:
        return ReverseSegmentOperator([best_pair]), {}
    else:
        # No improvement found, return an empty operator
        return None, {}