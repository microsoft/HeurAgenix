import os
import random
import time
from datetime import datetime
from src.problems.base.env import BaseEnv
from src.util.util import load_function
from src.problems.max_cut.components import InsertNodeOperator, InsertEdgeOperator, SwapOperator, DeleteOperator

# Cache for graph properties to avoid re-calculation
_GRAPH_THRESHOLD_CACHE = {}

def get_dynamic_threshold(env, data_name):
    """
    Calculates a dynamic quality threshold based on graph properties.
    - Positive graphs (NegRatio < 5%): High threshold (0.75)
    - Signed graphs: Threshold decreases with density (Dense signed graphs are harder)
    """
    if data_name in _GRAPH_THRESHOLD_CACHE:
        return _GRAPH_THRESHOLD_CACHE[data_name]
    
    node_num = env.instance_data["node_num"]
    adj = env.instance_data["adj"]
    
    edge_count = 0
    neg_edge_count = 0
    
    # Iterate adjacency list to count edges and negative weights
    # adj is a list of dicts: adj[u][v] = w
    for u in range(node_num):
        for v, w in adj[u].items():
            if u < v: # Count each undirected edge once
                edge_count += 1
                if w < 0:
                    neg_edge_count += 1
                    
    if edge_count == 0:
        threshold = 0.0
    else:
        neg_ratio = neg_edge_count / edge_count
        # Density = 2|E| / (|V|(|V|-1))
        density = 2 * edge_count / (node_num * (node_num - 1)) if node_num > 1 else 0
        
        if neg_ratio < 0.05:
            # Mostly positive graph - easier to get good initial solution
            threshold = 0.75
        else:
            # Signed graph - harder
            # Formula derived from log analysis:
            # G64 (Density ~0.0017) -> Needs ~0.48
            # G81 (Density ~0.0002) -> Needs ~0.60
            # Linear fit: Threshold = 0.62 - (Density * 80)
            threshold = 0.62 - (density * 80)
            
            # Clamp values to reasonable range [0.40, 0.65] for signed graphs
            threshold = max(0.40, min(0.65, threshold))
            
    print(f"Dynamic Threshold Analysis for {data_name}: Nodes={node_num}, Edges={edge_count}, NegRatio={neg_ratio:.2f}, Density={density:.5f} -> Threshold={threshold:.4f}", flush=True)
    _GRAPH_THRESHOLD_CACHE[data_name] = threshold
    return threshold

class PhasedSearchBestHyperHeuristic:
    def __init__(
        self,
        heuristic_pool: list[str],
        problem: str,
    ) -> None:
        self.heuristic_pool_names = heuristic_pool
        self.problem = problem
        
        self.constructive_heuristics = []
        self.improvement_heuristics = []
        self.perturbation_heuristics = []
        self.ruin_heuristics = []
        self.mutation_heuristics = []
        
        self._classify_heuristics()

    def _classify_heuristics(self):
        # Manual classification of heuristics
        # This allows for more precise control than automatic type-based classification
        
        constructive_names = {
            "balance_biased_edge_placement_9f22",
            "balanced_cut_21d5",
            "balanced_random_7f42",
            "heaviest_edge_seed_eb0d",
            "heavy_edge_matching_seed_edd5",
            "highest_delta_node_b31b",
            "highest_weight_edge_eb0d",
            "most_weight_neighbors_320c",
            "random_5c59",
            "semi_greedy_node_grasp_bf9a",
            "softmax_gain_insertion_76de",
            "spectral_seed_fiedler_51e0",
            "continuous_mean_field_batch", # Part 3: Batch CMF
            "balanced_random_batch", # Part 3: Batch Random
            "weighted_degree_batch", # Part 3: Batch Weighted Degree
        }
        
        improvement_names = {
            "cached_delta_flip_3cfd",
            "first_improvement_flip_7a32",
            "greedy_swap_5bb6",
            "k_block_swap_topk_589e",
            "majority_neighbor_flip_67a0",
            "multi_flip_threshold_fd21",
            "multi_swap_2_dbfe",
            "multi_swap_2_dbff",
            "single_flip_gain_5bb5",
            "tabu_node_flip_cae6",
            "two_node_joint_flip_590a",
        }
        
        ruin_names = {
            "low_contribution_bottom_delete_0b5f",
            "low_contribution_delete_worst_0b60",
            "random_delete_node_0b5f",
            "cluster_expansion_delete_bfs4",
        }

        mutation_names = {
            "simulated_annealing_ed14",
            "simulated_annealing_ed15",
        }

        for h_name in self.heuristic_pool_names:
            # Strip .py extension if present for matching
            base_name = os.path.basename(h_name).replace(".py", "")
            
            func = load_function(h_name, problem=self.problem)
            
            if base_name in constructive_names:
                self.constructive_heuristics.append(func)
            elif base_name in improvement_names:
                self.improvement_heuristics.append(func)
            elif base_name in ruin_names:
                self.ruin_heuristics.append(func)
                self.perturbation_heuristics.append(func) # Keep in general pool for backward compatibility if needed
            elif base_name in mutation_names:
                self.mutation_heuristics.append(func)
                self.perturbation_heuristics.append(func)
            else:
                print(f"Warning: Heuristic '{h_name}' not found in manual classification lists. Skipping.", flush=True)

    def run(self, env: BaseEnv) -> bool:
        current_steps = 0
        
        data = env.output_dir.split(os.sep)[-3]
        experiment = env.output_dir.split(os.sep)[-2]
        run_id = env.output_dir.split(os.sep)[-1]
        
        begin = datetime.now()
        last_value = 0
        found_best = False
        node_num = env.instance_data["node_num"]
        print(f"Start running phased search. Data:{data}\tExp\t{experiment}\tID:{run_id}\tStart:{begin.strftime('%Y-%m-%d %H:%M:%S')}\t", flush=True)
        
        # Optimization for Large Graphs (> 5000 nodes):
        # Use only O(1) or O(N) constructive heuristics to avoid O(N^2) bottlenecks.
        # This allows the search to reach the improvement/ruin phase much faster.
        active_constructive_heuristics = self.constructive_heuristics
        active_improvement_heuristics = self.improvement_heuristics
        
        if node_num > 5000:
            fast_constructive_names = {
                "random_5c59", 
                "balanced_random_7f42",
                "balanced_cut_21d5",
                "continuous_mean_field_batch", # New physics-inspired heuristic
                "balanced_random_batch",
                "weighted_degree_batch",
                "highest_delta_node_b31b", # Include slow ones for hybrid strategy?
                "most_weight_neighbors_320c",
                "softmax_gain_insertion_76de",
            }
            # Filter heuristics by name
            fast_heuristics = [h for h in self.constructive_heuristics if h.__name__ in fast_constructive_names]
            
            if fast_heuristics:
                print(f"Large graph detected ({node_num} nodes). Switching to Hybrid Constructive Heuristics.", flush=True)
                active_constructive_heuristics = fast_heuristics
                
                # Prioritize CMF if available
                cmf_heuristic = [h for h in fast_heuristics if h.__name__ == "continuous_mean_field_batch"]
                if cmf_heuristic:
                    # Give CMF a higher weight or make it the primary choice
                    # We can just duplicate it in the list to increase probability
                    active_constructive_heuristics.extend(cmf_heuristic * 5)
            else:
                print("Warning: Large graph detected but no fast heuristics found. Using default pool.", flush=True)

            # Also filter IMPROVEMENT heuristics for large graphs
            # We keep 'cached_delta_flip' for speed (it replaces slow greedy swaps).
            # We REMOVE 'tabu_node_flip' from the random pool because it is O(N^2) and too slow for frequent use.
            # Instead, we will trigger it conditionally when stuck.
            fast_improvement_names = {
                "cached_delta_flip_3cfd", # O(1) update, extremely fast greedy descent
                "first_improvement_flip_7a32", # O(N) scan, good for diversity
            }
            fast_imp_heuristics = [h for h in self.improvement_heuristics if h.__name__ in fast_improvement_names]
            
            # Extract Tabu for special use
            self.tabu_heuristic = next((h for h in self.improvement_heuristics if h.__name__ == "tabu_node_flip_cae6"), None)
            
            if fast_imp_heuristics:
                print(f"Large graph detected. Using optimized improvement set (Speed only): {[h.__name__ for h in fast_imp_heuristics]}", flush=True)
                print(f"Tabu heuristic '{self.tabu_heuristic.__name__ if self.tabu_heuristic else 'None'}' reserved for stagnation handling.", flush=True)
                active_improvement_heuristics = fast_imp_heuristics
            else:
                print("Warning: No optimized improvement heuristics found! Using full pool.", flush=True)
            self.tabu_heuristic = next((h for h in self.improvement_heuristics if h.__name__ == "tabu_node_flip_cae6"), None)
            
            if fast_imp_heuristics:
                print(f"Large graph detected. Using optimized improvement set (Speed): {[h.__name__ for h in fast_imp_heuristics]}", flush=True)
                if self.tabu_heuristic:
                    print("Tabu heuristic reserved for stagnation breaking.", flush=True)
                active_improvement_heuristics = fast_imp_heuristics
            else:
                print("Warning: No optimized improvement heuristics found! Using full pool.", flush=True)


        # Track stagnation
        no_improve_steps = 0
        max_no_improve = int(node_num * 2)  # Dynamic threshold based on problem size
        
        # Optimization for Large Graphs (> 5000 nodes):
        if node_num > 5000:
            # Increase patience for large graphs as operators might be slower but more impactful
            # Or decrease it if we want more frequent perturbations. 
            # For Tabu-like behavior, we want to explore local optima fully.
            max_no_improve = int(node_num * 4) 

        
        # Track perturbation cycles for massive ruin (Large Neighborhood Search)
        perturbation_count = 0
        max_perturbations_before_ruin = 5 # More patience before triggering massive ruin
        
        # Adaptive Ruin Parameters
        current_ruin_percent = 0.2
        best_at_last_ruin = 0

        current_best = 0
        
        # We rely on the Early Stopping mechanism (stagnation at max ruin) to terminate the run.
        # This allows the search to continue as long as it is making progress.
        while env.continue_run:
            
            # Phase 1: Construction
            if not env.is_complete_solution:
                if not active_constructive_heuristics:
                    print("Error: No constructive heuristics available but solution is incomplete.", flush=True)
                    break
                
                # Hierarchical Hybrid Construction Strategy
                # Mix Fast Batch Heuristics (High Prob) with Slow Precision Heuristics (Low Prob)
                
                # 1. Identify Batch vs Single Heuristics
                # We want to use it sparingly (20% prob), not frequently (80% prob).
                batch_heuristics = [h for h in active_constructive_heuristics if "batch" in h.__name__]
                single_heuristics = [h for h in active_constructive_heuristics if h not in batch_heuristics]
                
                # 2. Determine Strategy
                use_batch = False
                # Prefer batch if available, with 70% probability (Balanced Hybrid)
                # Lower probability (e.g. 0.7 vs 0.9) increases diversity by allowing more random/greedy single insertions.
                if batch_heuristics and (not single_heuristics or random.random() < 0.5):
                    use_batch = True
                
                # 3. Execute
                if use_batch:
                    heuristic = random.choice(batch_heuristics)
                    # Use ratio instead of fixed batch size
                    # 1% of nodes per batch allows for ~100 phases of construction (Fine-grained)
                    env.run_heuristic(heuristic, parameters={"batch_ratio": 0.05})
                else:
                    # Single Insertion (Precision)
                    if single_heuristics:
                        heuristic = random.choice(single_heuristics)
                        env.run_heuristic(heuristic)
                    elif batch_heuristics:
                        # Fallback: Use batch heuristic as single insertion
                        heuristic = random.choice(batch_heuristics)
                        env.run_heuristic(heuristic)
                
                current_steps += 1
                
                last_value = env.key_value
                selected_nodes = len(env.current_solution.set_a) + len(env.current_solution.set_b)
                end = datetime.now()
                time_cost = (end - begin).total_seconds()
                
                # Check if construction just finished
                if env.is_complete_solution and env.is_valid_solution and current_best == 0:
                    # === Quality Gate (Early Rejection) ===
                    # Strategy: "Kill Low Quality"
                    # If the constructed solution is too far from the best known,
                    # we assume it's in a bad basin of attraction and abort immediately.
                    
                    # Dynamic thresholds based on log analysis (20th percentile of good runs)
                    case_name = data.split('.')[0]
                    quality_threshold = get_dynamic_threshold(env, case_name)
                    
                    quality_ratio = env.key_value / env.best_known

                    if env.key_value > current_best:
                        current_best = env.key_value
                        selected_nodes = len(env.current_solution.set_a) + len(env.current_solution.set_b)
                        end = datetime.now()
                        time_cost = (end - begin).total_seconds()
                        init_value = env.key_value
                        print(f"Data:{data}\tExp:{experiment}\tID:{run_id}\tConstruction completed", flush=True)
                        print(f"Data:{data}\tExp:{experiment}\tID:{run_id}\tSteps:{current_steps}\tSelected:{selected_nodes}\tTotal:{node_num}\tInit:{init_value}\tNow:{env.key_value}\tCurrent best:{current_best}\tBest known:{env.best_known}\tNow:{end.strftime('%Y-%m-%d %H:%M:%S')}\tTime cost(hour):{time_cost/3600:.4f}", flush=True)
                        if quality_ratio < quality_threshold:
                            print(f"Data:{data}\tExp:{experiment}\tID:{run_id}\t [Quality Gate] Initial score {env.key_value} ({quality_ratio:.1%}) < {quality_threshold:.0%}. Aborting run to restart.", flush=True)
                            return False # Return False to signal the runner to stop this episode
                        else:
                            print(f"Data:{data}\tExp:{experiment}\tID:{run_id}\t [Quality Gate] Initial score {env.key_value} ({quality_ratio:.1%}) >= {quality_threshold:.0%}. Starting improvement phase.", flush=True)
                
            # Phase 2: Improvement (and Perturbation)
            else:
                # Check if we need perturbation
                if no_improve_steps > max_no_improve:
                    perturbation_count += 1
                    
                    # Check for Massive Ruin (Continuous Deletion)
                    if perturbation_count > max_perturbations_before_ruin:
                        
                        # Adaptive Logic: Did we improve since the last ruin?
                        if current_best > best_at_last_ruin:
                            # Yes, we improved! Reset ruin intensity.
                            print(f"Run:{run_id} Progress made ({best_at_last_ruin} -> {current_best}). Resetting ruin intensity.", flush=True)
                            current_ruin_percent = 0.2
                            best_at_last_ruin = current_best
                        else:
                            # No, we are stuck in the same basin. Increase intensity.
                            old_ruin = current_ruin_percent
                            current_ruin_percent = min(0.5, current_ruin_percent + 0.05)
                            print(f"Run:{run_id} No progress since last ruin. Intensifying ruin: {old_ruin:.2f} -> {current_ruin_percent:.2f}", flush=True)

                        # EARLY STOPPING: If we are at 50% ruin and still stuck, abandon this run.
                        # The worker will pick up a new run (new seed) from the queue.
                        if current_ruin_percent >= 0.5:
                            # If we have reached the best known solution, we should not give up.
                            # Instead, we reset the ruin intensity to continue searching (Extended Mode).
                            if current_best >= env.best_known:
                                print(f"Run:{run_id} Reached Best Known ({current_best}). Extending search resources (Resetting Ruin).", flush=True)
                                current_ruin_percent = 0.2
                                best_at_last_ruin = current_best
                            else:
                                print(f"Run:{run_id} STUCK at {current_best} despite max ruin. EARLY STOPPING to change seed.", flush=True)
                                break

                        print(f"Run:{run_id} Stagnated after {perturbation_count} perturbations. MASSIVE RUIN (Backtracking) with {current_ruin_percent:.0%}.", flush=True)
                        
                        # Determine how many nodes to remove
                        nodes_to_remove = max(10, int(node_num * current_ruin_percent))
                        
                        removed_count = 0
                        # Continuous deletion loop
                        for _ in range(nodes_to_remove * 2): # Safety factor 2x attempts
                            if removed_count >= nodes_to_remove:
                                break
                            
                            # FIX: Only use RUIN heuristics (DeleteOperator) for massive ruin
                            # Previously, mutation heuristics (SwapOperator) were mixed in, causing "fake ruin"
                            if self.ruin_heuristics:
                                heuristic = random.choice(self.ruin_heuristics)
                                env.run_heuristic(heuristic)
                                removed_count += 1
                                current_steps += 1
                            else:
                                print("Error: No ruin heuristics available for massive ruin!", flush=True)
                                break
                        
                        print(f"  -> Removed {removed_count} nodes. Rebuilding (Randomized Mode)...", flush=True)
                        perturbation_count = 0
                        no_improve_steps = 0
                        last_value = env.key_value 
                        continue

                    # Normal (Small) Perturbation
                    # Delete a few nodes (5-20) instead of just 1 to shake it up more
                    # For large graphs, we need stronger perturbation
                    base_perturb = max(20, int(node_num * 0.005)) # 0.5% of nodes (e.g. 100 for G81)
                    perturb_size = random.randint(base_perturb, base_perturb * 2)
                    
                    for _ in range(perturb_size):
                        # For small perturbation, we can mix mutation and ruin
                        if self.mutation_heuristics and random.random() < 0.5:
                             heuristic = random.choice(self.mutation_heuristics)
                        elif self.ruin_heuristics:
                             heuristic = random.choice(self.ruin_heuristics)
                        else:
                             heuristic = random.choice(self.perturbation_heuristics)
                             
                        env.run_heuristic(heuristic)
                    
                    no_improve_steps = 0 # Reset counter
                    last_value = env.key_value
                    # After perturbation, we might be incomplete, so next loop will go to Phase 1
                    continue

                # Normal Improvement
                if not active_improvement_heuristics:
                     # If no improvement heuristics, just stop or continue random construction (unlikely)
                     break
                
                # STRATEGIC TABU INJECTION for Large Graphs
                # If we are stagnating but not yet ready for perturbation, try Tabu to break free.
                # Trigger at 25%, 50%, 75% of max_no_improve
                if node_num > 5000 and hasattr(self, 'tabu_heuristic') and self.tabu_heuristic:
                    thresholds = [int(max_no_improve * 0.25), int(max_no_improve * 0.5), int(max_no_improve * 0.75)]
                    if no_improve_steps in thresholds:
                        # print(f"Run:{run_id} Stagnation detected ({no_improve_steps}/{max_no_improve}). Injecting Tabu Search.")
                        env.run_heuristic(self.tabu_heuristic)
                        current_steps += 1
                        # Check if Tabu helped
                        if env.key_value > last_value:
                            last_value = env.key_value
                            no_improve_steps = 0
                            if env.is_valid_solution and env.key_value > current_best:
                                current_best = env.key_value
                                # Log update...
                        else:
                            no_improve_steps += 1
                        continue # Skip normal improvement this step

                heuristic = random.choice(active_improvement_heuristics)
                env.run_heuristic(heuristic)
                
                current_steps += 1
                
                # Check improvement
                if env.key_value > last_value:
                    last_value = env.key_value
                    no_improve_steps = 0
                    if env.is_valid_solution and env.key_value > current_best:
                        current_best = env.key_value
                        selected_nodes = len(env.current_solution.set_a) + len(env.current_solution.set_b)
                        end = datetime.now()
                        time_cost = (end - begin).total_seconds()
                        print(f"Data:{data}\tExp:{experiment}\tID:{run_id}\tSteps:{current_steps}\tSelected:{selected_nodes}\tTotal:{node_num}\tInit:{init_value}\tNow:{env.key_value}\tCurrent best:{current_best}\tBest known:{env.best_known}\tNow:{end.strftime('%Y-%m-%d %H:%M:%S')}\tTime cost(hour):{time_cost/3600:.4f}", flush=True)
                else:
                    no_improve_steps += 1

                # Logging
                current_best = max(current_best, env.key_value)

                if env.key_value == env.best_known:
                    if env.is_complete_solution and env.is_valid_solution:
                        env.dump_result(result_file=f"match_best_known_result.{experiment}.{run_id}.txt")
                        found_best = True
                        # Don't stop, try to improve more!
                        env.best_known = env.key_value # Update local best known to keep pushing

                # Check best known
                if env.key_value > env.best_known:
                    if env.is_complete_solution and env.is_valid_solution:
                        print(f"!!! NEW BEST FOUND: {env.key_value} > {env.best_known} !!!", flush=True)
                        print(env.current_solution, flush=True)
                        env.dump_result(result_file=f"break_best_known_result.{experiment}.{run_id}.txt")
                        found_best = True
                        # Don't stop, try to improve more!
                        env.best_known = env.key_value # Update local best known to keep pushing

        return found_best
