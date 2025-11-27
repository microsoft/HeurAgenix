from src.problems.max_cut.components import *
import numpy as np
import random

def tabu_node_flip_cae6(
    problem_state: dict,
    algorithm_data: dict,
    tabu_tenure: int = -1,
    aspiration: bool = True,
    aspiration_margin: float = 1e-12,
    allow_non_improving: bool = True,
    break_ties_randomly: bool = False,
    **kwargs
) -> tuple[SwapOperator, dict]:
    """
    Tabu node flip local search with aspiration and deterministic tie-breaking for MaxCut.
    Evaluates single-vertex flips (A→B or B→A) using a delta-gain model and returns a SwapOperator
    that flips exactly one node. A tabu list prevents cycling by disallowing recently flipped nodes for a
    fixed tenure; aspiration permits tabu moves if they improve the best-so-far cut. Nodes are scanned
    in a deterministic order (sorted(A) then sorted(B)); optional random tie-breaking diversifies choices.

    Key characteristics unique to this heuristic:
        - Delta model: For i in A, Δ = sum_w(i,A) − sum_w(i,B); for i in B, Δ = sum_w(i,B) − sum_w(i,A).
          For undirected MaxCut (symmetric weights), Δ equals the change in cut from flipping i.
        - Validity: Only currently assigned nodes (in A or B) are considered; unassigned nodes are ignored.
        - Tabu with aspiration: A node is tabu if its expiry > current iteration; aspiration allows the move
          if it strictly improves the historical best cut (by aspiration_margin).
        - Determinism with optional diversification: Fixed scan order avoids nondeterminism; ties can be broken
          randomly when break_ties_randomly=True.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): Symmetric n×n weight matrix for the undirected graph.
            - "current_solution" (Solution): Current partition with sets current_solution.set_a and current_solution.set_b.
            - "current_cut_value" (int or float): Current cut value used to evaluate predicted improvement.
            - "node_num" (int, optional): Number of nodes; if absent, inferred from weight_matrix.shape[0].
            - Other keys present in problem_state: Not used in this heuristic.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary / Not used in this heuristic:
            - "tabu" (dict[int, int], optional): Mapping node → expiry_iteration. Nodes with expiry > current iteration are tabu.
            - "iteration" (int, optional): Current iteration counter. Defaults to 0 if absent.
            - "best_cut" (float, optional): Best cut value observed so far. Defaults to current_cut_value if absent.
            - "tabu_tenure" (int, optional): Persisted tenure from previous calls; used if the function parameter tabu_tenure is -1.
        Hyper-parameters:
            - tabu_tenure (int): Number of iterations a flipped node remains tabu. Default is -1, which means:
                use algorithm_data["tabu_tenure"] if available; otherwise compute max(3, int(0.1 * n)).
            - aspiration (bool): Enable aspiration to allow tabu moves that strictly improve the best cut. Default is True.
            - aspiration_margin (float): Required strict improvement margin for aspiration (epsilon). Default is 1e-12.
            - allow_non_improving (bool): Permit returning a non-tabu move even if Δ ≤ 0 (diversification). Default is True.
            - break_ties_randomly (bool): Randomize among equal-Δ candidates to diversify choices. Default is False.

    Returns:
        SwapOperator: Flips one selected node to the opposite set (A↔B), preserving solution validity.
        dict: Updated algorithm data dictionary including:
            - "tabu": Pruned and updated tabu entries with the moved node's new expiry.
            - "iteration": Incremented iteration count.
            - "tabu_tenure": The tenure used in this call.
            - "best_cut": Updated if the predicted new cut improves the historical best.

        If no admissible move exists (e.g., no assigned nodes; or all moves are tabu without aspiration and allow_non_improving=False),
        returns (None, {}), which should be rare under default settings.
    """
    # Extract required problem state
    weight_matrix = problem_state["weight_matrix"]
    current_solution = problem_state["current_solution"]
    current_cut_value = problem_state["current_cut_value"]

    # Determine number of nodes
    n = int(problem_state.get("node_num", weight_matrix.shape[0]))
    if n <= 0:
        return None, {}

    # Resolve tabu tenure
    if tabu_tenure == -1:
        persisted_tenure = algorithm_data.get("tabu_tenure", None)
        default_tenure = max(3, int(0.1 * n))
        tabu_tenure = int(persisted_tenure) if persisted_tenure is not None else default_tenure

    # Pull state from algorithm_data
    iteration = int(algorithm_data.get("iteration", 0))
    tabu = dict(algorithm_data.get("tabu", {}))
    best_cut = float(algorithm_data.get("best_cut", current_cut_value))

    # Assigned nodes in deterministic order
    set_a = set(current_solution.set_a)
    set_b = set(current_solution.set_b)
    if not set_a and not set_b:
        return None, {}
    assigned_nodes = list(sorted(set_a)) + list(sorted(set_b))

    # Precompute side weights
    weight_to_a = weight_matrix[:, list(set_a)].sum(axis=1) if set_a else np.zeros(n, dtype=float)
    weight_to_b = weight_matrix[:, list(set_b)].sum(axis=1) if set_b else np.zeros(n, dtype=float)

    def is_tabu(node: int, curr_iter: int) -> bool:
        expiry = tabu.get(node, None)
        return expiry is not None and expiry > curr_iter

    best_non_tabu_node = None
    best_non_tabu_delta = -float("inf")
    best_asp_node = None
    best_asp_delta = -float("inf")

    # Evaluate all assigned nodes
    for node in assigned_nodes:
        if node in set_a:
            delta = float(weight_to_a[node] - weight_to_b[node])
        elif node in set_b:
            delta = float(weight_to_b[node] - weight_to_a[node])
        else:
            continue

        if not is_tabu(node, iteration):
            if delta > best_non_tabu_delta:
                best_non_tabu_delta = delta
                best_non_tabu_node = node
            elif break_ties_randomly and delta == best_non_tabu_delta and best_non_tabu_node is not None:
                if random.random() < 0.5:
                    best_non_tabu_delta = delta
                    best_non_tabu_node = node
        else:
            if aspiration:
                predicted_new_cut = current_cut_value + delta
                if predicted_new_cut > best_cut + aspiration_margin:
                    if delta > best_asp_delta:
                        best_asp_delta = delta
                        best_asp_node = node
                    elif break_ties_randomly and delta == best_asp_delta and best_asp_node is not None:
                        if random.random() < 0.5:
                            best_asp_delta = delta
                            best_asp_node = node

    # Choose move: aspiration preferred; else best non-tabu (optionally require improvement)
    chosen_node = None
    chosen_delta = 0.0

    if best_asp_node is not None:
        chosen_node = best_asp_node
        chosen_delta = best_asp_delta
    elif best_non_tabu_node is not None:
        if allow_non_improving or best_non_tabu_delta > 0.0:
            chosen_node = best_non_tabu_node
            chosen_delta = best_non_tabu_delta

    if chosen_node is None:
        return None, {}

    # Build operator
    op = SwapOperator([chosen_node])

    # Update algorithm data
    new_tabu = {nd: exp for nd, exp in tabu.items() if exp > iteration}
    new_tabu[chosen_node] = iteration + tabu_tenure

    new_iteration = iteration + 1
    predicted_new_cut = current_cut_value + chosen_delta
    new_best_cut = best_cut if predicted_new_cut <= best_cut else predicted_new_cut

    updated_algo_data = {
        "tabu": new_tabu,
        "iteration": new_iteration,
        "tabu_tenure": tabu_tenure,
        "best_cut": new_best_cut,
    }

    return op, updated_algo_data