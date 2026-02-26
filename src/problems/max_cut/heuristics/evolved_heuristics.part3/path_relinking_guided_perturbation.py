import random
from src.problems.max_cut.components import Solution, SwapOperator

def path_relinking_guided_perturbation(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[SwapOperator, dict]:
    """
    Path Relinking Guided Perturbation.
    
    Instead of random perturbation, this heuristic moves the current solution 
    towards a random target solution from the 'elite_pool'.
    
    This acts as a 'smart' jump out of a local optimum, exploring the path 
    between two high-quality solutions (Path Relinking), where new local optima 
    often reside.
    """
    current_solution = problem_state.get("current_solution")
    elite_pool = algorithm_data.get("elite_pool", [])
    
    if not current_solution or not elite_pool:
        return None, {}
    
    # 1. Select Target: "Distant Elite Strategy"
    # We want to find a target that is HIGH QUALITY but STRUCTURALLY DIFFERENT.
    # This promotes "Ridge Walking" rather than just climbing the same peak.
    
    # Filter reasonable candidates (e.g. within top 20% of pool or > some threshold)
    # If possible, filter those > 99.5% of Best Known to ensure quality.
    candidates = [s for s in elite_pool if s.cut_value >= current_solution.cut_value * 0.99]
    if not candidates:
        candidates = elite_pool
        
    # Function to calculate hamming distance (approximate or exact)
    def calc_dist(sol1, sol2):
        # Orientation 1
        d1 = len((sol1.set_a & sol2.set_b) | (sol1.set_b & sol2.set_a))
        # Orientation 2
        d2 = len((sol1.set_a & sol2.set_a) | (sol1.set_b & sol2.set_b))
        return min(d1, d2)
        
    if candidates:
        # Sample a subset to avoid O(N) distance checks if pool is huge
        sample_size = min(len(candidates), 20)
        sample = random.sample(candidates, sample_size)
        
        # Pick the one with MAX distance from current
        # Target = argmax_s (Distance(Current, s))
        # Logic: If we are at peak A, and peak B is far away, the path A->B covers new ground.
        target_solution = max(sample, key=lambda s: calc_dist(current_solution, s))
    else:
        target_solution = random.choice(elite_pool)
    
    # 2. Calculate Symmetric Difference (nodes with different assignments)
    # Target structure: Solution(set_a, set_b)
    # Set A in current vs Set A in target.
    # Note: MaxCut solution is symmetric (A, B) == (B, A). 
    # We must check both orientations to find the minimum difference (Hamming distance).
    
    # Orientation 1: A->A, B->B
    # Discrepancy: Nodes in (CurA - TarA) U (CurA - TarB)? No.
    # Nodes where Cur(u) != Tar(u).
    # i.e. (CurA & TarB) U (CurB & TarA)
    
    diff_1 = (current_solution.set_a & target_solution.set_b) | (current_solution.set_b & target_solution.set_a)
    
    # Orientation 2: A->B, B->A (Flip Target)
    # i.e. (CurA & TarA) U (CurB & TarB)
    diff_2 = (current_solution.set_a & target_solution.set_a) | (current_solution.set_b & target_solution.set_b)
    
    if len(diff_1) <= len(diff_2):
        diff_nodes = list(diff_1)
    else:
        diff_nodes = list(diff_2)
        
    if not diff_nodes:
        return None, {}
    
    # 3. Determine Step Size based on 'intensity' parameter
    # intensity 0.0 -> 1.0. 
    # 0.5 means move 50% of the way (Hamming distance / 2).
    intensity = kwargs.get("intensity", 0.3) 
    
    move_count = max(1, int(len(diff_nodes) * intensity))
    
    # 4. Select nodes to flip
    # Random selection from the difference set
    nodes_to_swap = random.sample(diff_nodes, move_count)
    
    return SwapOperator(nodes=nodes_to_swap), {}
