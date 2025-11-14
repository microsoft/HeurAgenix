from src.problems.max_cut.components import *
def degree_product_edge_chooser_97c3(problem_state: dict, algorithm_data: dict, top_k: int=0, use_abs: bool=False, ignore_diagonal: bool=True, **kwargs) -> tuple[InsertEdgeOperator, dict]:
    """Degree-product oriented edge insertion with orientation by local cut gain.
    
    This constructive heuristic selects two currently unassigned vertices by maximizing the product of their weighted degrees, biasing toward structurally central endpoints without scanning all pairwise edge weights. After selecting the pair, it orients them into opposite sets (A/B) based on which orientation yields the larger immediate increase in the cut value relative to the existing partition. If only one unassigned vertex remains (or top_k restriction leaves fewer than two candidates), it inserts that single vertex into the side with higher immediate gain.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): Symmetric adjacency/weight matrix of the undirected graph (shape: node_num x node_num).
            - "current_solution" (Solution): Current partition with attributes set_a (set[int]) and set_b (set[int]).
            - "unselected_nodes" (set[int]): Nodes not yet placed in either set; candidates for insertion.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this heuristic.
        top_k (int): If > 0, restrict candidate nodes to the top_k unselected nodes by weighted degree (descending) before pairing. Reduces O(|U|^2) to O(top_k^2); ignored when < 2. Default is 0.
        use_abs (bool): If True, compute degrees using absolute weights (sum |w(i,·)|), helpful when negative weights exist to reflect magnitude-based centrality. Default is False.
        ignore_diagonal (bool): If True, exclude any self-loop weight w(i,i) from degree computation. Default is True.

    Returns:
        InsertEdgeOperator: Operator that inserts the chosen oriented pair (node_1→A, node_2→B) to maximize immediate cut gain. 
            - Boundary case: If fewer than two candidates remain, returns an InsertNodeOperator for the single node with the better immediate gain.
        dict: Empty dictionary; this heuristic does not update algorithm_data.

    """
    weight_matrix = problem_state["weight_matrix"]
    current_solution = problem_state["current_solution"]
    unselected_nodes = problem_state["unselected_nodes"]

    # If there are no unselected nodes, no constructive move is possible.
    if not unselected_nodes:
        return None, {}

    # If exactly one node remains, insert it to the side with larger immediate gain.
    if len(unselected_nodes) == 1:
        lone = next(iter(unselected_nodes))
        gain_to_A = sum(weight_matrix[lone][b] for b in current_solution.set_b) if current_solution.set_b else 0
        gain_to_B = sum(weight_matrix[lone][a] for a in current_solution.set_a) if current_solution.set_a else 0
        target_set = 'A' if gain_to_A >= gain_to_B else 'B'
        return InsertNodeOperator(node=lone, target_set=target_set), {}

    # Compute weighted degrees for all unselected nodes.
    def node_degree(i: int) -> float:
        row = weight_matrix[i]
        total = 0.0
        for j in range(len(row)):
            if ignore_diagonal and j == i:
                continue
            w = row[j]
            total += abs(w) if use_abs else w
        return total

    deg_list = [(u, node_degree(u)) for u in unselected_nodes]

    # Optionally restrict to top_k by degree.
    if isinstance(top_k, int) and top_k > 1:
        deg_list.sort(key=lambda x: x[1], reverse=True)
        deg_list = deg_list[:min(top_k, len(deg_list))]

    # If restriction leaves fewer than two candidates, insert the single best node.
    if len(deg_list) < 2:
        best_node = deg_list[0][0]
        gain_to_A = sum(weight_matrix[best_node][b] for b in current_solution.set_b) if current_solution.set_b else 0
        gain_to_B = sum(weight_matrix[best_node][a] for a in current_solution.set_a) if current_solution.set_a else 0
        target_set = 'A' if gain_to_A >= gain_to_B else 'B'
        return InsertNodeOperator(node=best_node, target_set=target_set), {}

    # Select the pair with maximal degree product.
    nodes = [n for n, _ in deg_list]
    deg_map = {n: d for n, d in deg_list}
    best_pair = None
    best_product = float('-inf')

    for idx_i in range(len(nodes)):
        i = nodes[idx_i]
        di = deg_map[i]
        for idx_j in range(idx_i + 1, len(nodes)):
            j = nodes[idx_j]
            dj = deg_map[j]
            prod = di * dj
            if prod > best_product:
                best_product = prod
                best_pair = (i, j)

    if best_pair is None:
        return None, {}

    i, j = best_pair

    # Orient the chosen pair by immediate cut gain relative to the current partition.
    gain_i_to_A = sum(weight_matrix[i][b] for b in current_solution.set_b) if current_solution.set_b else 0
    gain_j_to_B = sum(weight_matrix[j][a] for a in current_solution.set_a) if current_solution.set_a else 0
    delta_a_to_b = gain_i_to_A + gain_j_to_B + weight_matrix[i][j]

    gain_i_to_B = sum(weight_matrix[i][a] for a in current_solution.set_a) if current_solution.set_a else 0
    gain_j_to_A = sum(weight_matrix[j][b] for b in current_solution.set_b) if current_solution.set_b else 0
    delta_b_to_a = gain_i_to_B + gain_j_to_A + weight_matrix[i][j]

    operator = InsertEdgeOperator(node_1=i, node_2=j) if delta_a_to_b >= delta_b_to_a else InsertEdgeOperator(node_1=j, node_2=i)
    return operator, {}