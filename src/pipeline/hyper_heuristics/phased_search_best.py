import os
import random
import time
from datetime import datetime
from src.problems.base.env import BaseEnv
from src.util.util import load_function
from src.problems.max_cut.components import InsertNodeOperator, InsertEdgeOperator, SwapOperator, DeleteOperator

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
                print(f"Warning: Heuristic '{h_name}' not found in manual classification lists. Skipping.")

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
                print(f"Large graph detected ({node_num} nodes). Switching to Hybrid Constructive Heuristics.")
                active_constructive_heuristics = fast_heuristics
                
                # Prioritize CMF if available
                cmf_heuristic = [h for h in fast_heuristics if h.__name__ == "continuous_mean_field_batch"]
                if cmf_heuristic:
                    # Give CMF a higher weight or make it the primary choice
                    # We can just duplicate it in the list to increase probability
                    active_constructive_heuristics.extend(cmf_heuristic * 5)
            else:
                print("Warning: Large graph detected but no fast heuristics found. Using default pool.")

            # Also filter IMPROVEMENT heuristics for large graphs
            # We keep 'cached_delta_flip' for speed (it replaces slow greedy swaps).
            # We KEEP 'tabu_node_flip' because it is essential for breaking local optima, even if slow.
            fast_improvement_names = {
                "cached_delta_flip_3cfd", # O(1) update, extremely fast greedy descent
                "first_improvement_flip_7a32", # O(N) scan, good for diversity
                "tabu_node_flip_cae6" # Essential for Gset record breaking. Slower (O(N^2) without cache), but worth it.
            }
            fast_imp_heuristics = [h for h in self.improvement_heuristics if h.__name__ in fast_improvement_names]
            if fast_imp_heuristics:
                print(f"Large graph detected. Using optimized improvement set (Speed + Tabu): {[h.__name__ for h in fast_imp_heuristics]}")
                active_improvement_heuristics = fast_imp_heuristics
            else:
                print("Warning: No optimized improvement heuristics found! Using full pool.")


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
                    print("Error: No constructive heuristics available but solution is incomplete.")
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
                    # If the constructed solution is too far from the best known (e.g. < 75%),
                    # we assume it's in a bad basin of attraction and abort immediately.
                    # This frees up the worker to try a new random seed.
                    quality_threshold = 0.75 # 75% of best known. For G81 (14060) -> 10545
                    quality_ratio = env.key_value / env.best_known

                    if env.key_value > current_best:
                        current_best = env.key_value
                        selected_nodes = len(env.current_solution.set_a) + len(env.current_solution.set_b)
                        end = datetime.now()
                        time_cost = (end - begin).total_seconds()
                        init_value = env.key_value
                        print(f"Data:{data}\tExp:{experiment}\tID:{run_id}\tConstruction completed")
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
                            print(f"Run:{run_id} Progress made ({best_at_last_ruin} -> {current_best}). Resetting ruin intensity.")
                            current_ruin_percent = 0.2
                            best_at_last_ruin = current_best
                        else:
                            # No, we are stuck in the same basin. Increase intensity.
                            old_ruin = current_ruin_percent
                            current_ruin_percent = min(0.5, current_ruin_percent + 0.05)
                            print(f"Run:{run_id} No progress since last ruin. Intensifying ruin: {old_ruin:.2f} -> {current_ruin_percent:.2f}")

                        # EARLY STOPPING: If we are at 50% ruin and still stuck, abandon this run.
                        # The worker will pick up a new run (new seed) from the queue.
                        if current_ruin_percent >= 0.5:
                            print(f"Run:{run_id} STUCK at {current_best} despite max ruin. EARLY STOPPING to change seed.")
                            break

                        print(f"Run:{run_id} Stagnated after {perturbation_count} perturbations. MASSIVE RUIN (Backtracking) with {current_ruin_percent:.0%}.")
                        
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
                                print("Error: No ruin heuristics available for massive ruin!")
                                break
                        
                        print(f"  -> Removed {removed_count} nodes. Rebuilding (Randomized Mode)...")
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
                        env.dump_result(result_file=f"match_best_known_result.txt")
                        found_best = True
                        # Don't stop, try to improve more!
                        env.best_known = env.key_value # Update local best known to keep pushing

                # Check best known
                if env.key_value > env.best_known:
                    if env.is_complete_solution and env.is_valid_solution:
                        print(f"!!! NEW BEST FOUND: {env.key_value} > {env.best_known} !!!", flush=True)
                        print(env.current_solution, flush=True)
                        env.dump_result(result_file=f"break_best_known_result_test_only.txt")
                        found_best = True
                        # Don't stop, try to improve more!
                        env.best_known = env.key_value # Update local best known to keep pushing

        return found_best
