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
    break_ties_randomly: bool = True,
    steps: int = 1000,
    **kwargs
) -> tuple[SwapOperator, dict]:
    """
    Tabu node flip local search with aspiration and deterministic tie-breaking for MaxCut.
    Supports 'Burst Mode' via the `steps` parameter to perform multiple iterations efficiently.
    """
    # Extract required problem state
    weight_matrix = problem_state["weight_matrix"]
    current_solution = problem_state["current_solution"]
    current_cut_value = problem_state["current_cut_value"]
    adj = problem_state.get("adj") # Required for efficient updates

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

    # Initialize Sets and Partition Array
    set_a = set(current_solution.set_a)
    set_b = set(current_solution.set_b)
    
    # Partition array: 0 for A, 1 for B (for easy indexing)
    partition = np.zeros(n, dtype=int)
    if set_b:
        partition[list(set_b)] = 1
    
    # Precompute side weights (Expensive, done once)
    # weight_to_a[i] = sum of weights from i to nodes in A
    weight_to_a = weight_matrix[:, list(set_a)].sum(axis=1) if set_a else np.zeros(n, dtype=float)
    weight_to_b = weight_matrix[:, list(set_b)].sum(axis=1) if set_b else np.zeros(n, dtype=float)

    flipped_nodes_tracker = set()
    current_val = current_cut_value

    for _ in range(steps):
        # Calculate Gains
        # If i in A (partition[i]==0): Gain = w(i, A) - w(i, B) = weight_to_a - weight_to_b
        # If i in B (partition[i]==1): Gain = w(i, B) - w(i, A) = weight_to_b - weight_to_a
        # So Gain = (weight_to_a - weight_to_b) * (1 if A else -1)
        
        diff = weight_to_a - weight_to_b
        signs = 1 - 2 * partition
        gains = diff * signs
        
        # Identify Candidates
        is_tabu_mask = np.zeros(n, dtype=bool)
        current_iter_limit = iteration
        for t_node, t_expiry in tabu.items():
            if t_expiry > current_iter_limit:
                is_tabu_mask[t_node] = True
                
        # Best Non-Tabu
        masked_gains = np.copy(gains)
        masked_gains[is_tabu_mask] = -float("inf")
        
        best_non_tabu_idx = np.argmax(masked_gains)
        best_non_tabu_val = masked_gains[best_non_tabu_idx]
        
        # Best Aspiration (Global Best)
        best_overall_idx = np.argmax(gains)
        best_overall_val = gains[best_overall_idx]
        
        chosen_node = None
        chosen_delta = 0.0
        
        # Check Aspiration
        if best_overall_val > -float("inf"):
            predicted_val = current_val + best_overall_val
            if predicted_val > best_cut + aspiration_margin:
                chosen_node = best_overall_idx
                chosen_delta = best_overall_val
        
        # If not aspiration, use non-tabu
        if chosen_node is None and best_non_tabu_val > -float("inf"):
            if allow_non_improving or best_non_tabu_val > 0:
                chosen_node = best_non_tabu_idx
                chosen_delta = best_non_tabu_val
        
        if chosen_node is None:
            break # No valid move
            
        # Execute Flip
        # 1. Update Solution Sets (Locally)
        if partition[chosen_node] == 0: # A -> B
            partition[chosen_node] = 1
            set_a.remove(chosen_node)
            set_b.add(chosen_node)
        else: # B -> A
            partition[chosen_node] = 0
            set_b.remove(chosen_node)
            set_a.add(chosen_node)
            
        # 2. Update Weights (Incremental)
        neighbors = adj[chosen_node]
        for v, w in neighbors.items():
            if partition[chosen_node] == 1: # Moved to B
                weight_to_a[v] -= w
                weight_to_b[v] += w
            else: # Moved to A
                weight_to_a[v] += w
                weight_to_b[v] -= w
                
        # 3. Update Tabu
        tabu[chosen_node] = iteration + tabu_tenure
        
        # 4. Update Global State
        current_val += chosen_delta
        if current_val > best_cut:
            best_cut = current_val
            
        iteration += 1
        
        # Track Flip
        if chosen_node in flipped_nodes_tracker:
            flipped_nodes_tracker.remove(chosen_node)
        else:
            flipped_nodes_tracker.add(chosen_node)

    # End Loop
    
    # Build Operator
    if not flipped_nodes_tracker:
        return None, {}
        
    op = SwapOperator(list(flipped_nodes_tracker))
    
    # Clean Tabu
    new_tabu = {nd: exp for nd, exp in tabu.items() if exp > iteration}
    
    updated_algo_data = {
        "tabu": new_tabu,
        "iteration": iteration,
        "tabu_tenure": tabu_tenure,
        "best_cut": best_cut,
    }
    
    return op, updated_algo_data
