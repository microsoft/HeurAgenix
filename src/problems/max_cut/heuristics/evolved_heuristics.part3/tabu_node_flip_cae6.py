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
    Optimized with vectorized operations for large graphs.
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

    # Initialize Sets and Partition Array
    set_a = set(current_solution.set_a)
    set_b = set(current_solution.set_b)
    
    # Partition array: 0 for A, 1 for B
    partition = np.zeros(n, dtype=int)
    if set_b:
        partition[list(set_b)] = 1
    
    # Precompute side weights
    weight_to_a = weight_matrix[:, list(set_a)].sum(axis=1) if set_a else np.zeros(n, dtype=float)
    weight_to_b = weight_matrix[:, list(set_b)].sum(axis=1) if set_b else np.zeros(n, dtype=float)

    # Precompute Gains
    # If i in A (0): Gain = w(i, A) - w(i, B) = weight_to_a - weight_to_b
    # If i in B (1): Gain = w(i, B) - w(i, A) = weight_to_b - weight_to_a
    # signs: 1 for A, -1 for B
    signs = 1 - 2 * partition
    diff = weight_to_a - weight_to_b
    gains = diff * signs

    flipped_nodes_tracker = set()
    current_val = current_cut_value
    
    # Track the best state found DURING this call
    best_val_in_run = current_val
    best_flipped_tracker = set()
    
    # Tabu array for faster lookup (instead of dict)
    tabu_expiry = np.zeros(n, dtype=int)
    for node, expiry in tabu.items():
        if node < n:
            tabu_expiry[node] = expiry

    for _ in range(steps):
        # Identify Candidates
        # Mask tabu nodes
        is_tabu = tabu_expiry > iteration
        
        # Best Non-Tabu
        masked_gains = np.copy(gains)
        masked_gains[is_tabu] = -float("inf")
        
        best_non_tabu_idx = np.argmax(masked_gains)
        best_non_tabu_val = masked_gains[best_non_tabu_idx]
        
        # Best Overall (for Aspiration)
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
        chosen_node = int(chosen_node) # CAST TO INT TO AVOID KEYERROR WITH NUMPY TYPES
        
        # 1. Update Solution Sets (Locally)
        old_partition = partition[chosen_node]
        new_partition = 1 - old_partition
        partition[chosen_node] = new_partition
        
        if old_partition == 0: # A -> B
            if chosen_node in set_a:
                set_a.remove(chosen_node)
            set_b.add(chosen_node)
        else: # B -> A
            if chosen_node in set_b:
                set_b.remove(chosen_node)
            set_a.add(chosen_node)
            
        # 2. Update Weights & Gains (Vectorized)
        # Get neighbors and their weights
        neighbors_mask = weight_matrix[chosen_node] != 0
        weights = weight_matrix[chosen_node, neighbors_mask]
        neighbor_indices = np.where(neighbors_mask)[0]
        
        if new_partition == 1: # Moved to B
            weight_to_a[neighbor_indices] -= weights
            weight_to_b[neighbor_indices] += weights
            # Update gains for neighbors
            gains[neighbor_indices] -= 2 * weights * signs[neighbor_indices]
        else: # Moved to A
            weight_to_a[neighbor_indices] += weights
            weight_to_b[neighbor_indices] -= weights
            gains[neighbor_indices] += 2 * weights * signs[neighbor_indices]
            
        # Update own gain and sign
        signs[chosen_node] = -signs[chosen_node]
        gains[chosen_node] = -gains[chosen_node]
        
        # 3. Update Tabu
        tabu_expiry[chosen_node] = iteration + tabu_tenure
        
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

        # Update best tracking
        if current_val > best_val_in_run:
            best_val_in_run = current_val
            best_flipped_tracker = flipped_nodes_tracker.copy()

    # End Loop
    
    # Build Operator based on BEST state, not FINAL state
    if not best_flipped_tracker:
        # If no improvement found, return empty (or None)
        # But wait, if we found NOTHING better than start, best_flipped_tracker is empty.
        # This is correct behavior for a hill climber (step returns no improvement).
        return None, {}
        
    op = SwapOperator(list(best_flipped_tracker))
    
    # Convert tabu array back to dict for persistence (only active ones)
    new_tabu = {}
    active_tabu_indices = np.where(tabu_expiry > iteration)[0]
    for idx in active_tabu_indices:
        new_tabu[int(idx)] = int(tabu_expiry[idx])
    
    updated_algo_data = {
        "tabu": new_tabu,
        "iteration": iteration,
        "tabu_tenure": tabu_tenure,
        "best_cut": best_cut,
    }
    
    return op, updated_algo_data
