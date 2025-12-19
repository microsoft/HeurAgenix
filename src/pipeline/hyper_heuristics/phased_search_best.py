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
        
        perturbation_names = {
            "low_contribution_bottom_delete_0b5f",
            "low_contribution_delete_worst_0b60",
            "random_delete_node_0b5f",
            "simulated_annealing_ed14",
            "simulated_annealing_ed15",
        }

        for h_name in self.heuristic_pool_names:
            # Strip .py extension if present for matching
            base_name = h_name.replace(".py", "")
            
            func = load_function(h_name, problem=self.problem)
            
            if base_name in constructive_names:
                self.constructive_heuristics.append(func)
            elif base_name in improvement_names:
                self.improvement_heuristics.append(func)
            elif base_name in perturbation_names:
                self.perturbation_heuristics.append(func)
            else:
                print(f"Warning: Heuristic '{h_name}' not found in manual classification lists. Skipping.")

    def run(self, env: BaseEnv) -> bool:
        current_steps = 0
        
        data = env.output_dir.split(os.sep)[-3]
        experiment = env.output_dir.split(os.sep)[-2]
        run_id = env.output_dir.split(os.sep)[-1]
        
        print(f"start running phased search: {data}, {experiment}, {run_id}", flush=True)
        
        begin = datetime.now()
        last_value = 0
        found_best = False
        node_num = env.instance_data["node_num"]
        
        # Track stagnation
        no_improve_steps = 0
        max_no_improve = 200  # Reduced threshold for faster reaction
        
        # Track tried heuristics for immediate stagnation detection
        tried_heuristics = set()
        
        # Track perturbation cycles for massive ruin (Large Neighborhood Search)
        perturbation_count = 0
        max_perturbations_before_ruin = 5 # Reduced to trigger massive ruin sooner
        
        # Adaptive Ruin Parameters
        current_ruin_percent = 0.2
        best_at_last_ruin = 0
        rebuilding_mode = False  # Flag to indicate we are rebuilding after a massive ruin

        current_best = 0
        
        # We rely on the Early Stopping mechanism (stagnation at max ruin) to terminate the run.
        # This allows the search to continue as long as it is making progress.
        while env.continue_run:
            
            # Phase 1: Construction
            if not env.is_complete_solution:
                if not self.constructive_heuristics:
                    print("Error: No constructive heuristics available but solution is incomplete.")
                    break
                
                # Interleaved Optimization: Small chance to run improvement during construction
                assigned_nodes = len(env.current_solution.set_a) + len(env.current_solution.set_b)
                if self.improvement_heuristics and assigned_nodes > 20 and random.random() < 0.1:
                    # During rebuilding, we might want to avoid greedy improvements too early, 
                    # but interleaved optimization is generally good.
                    heuristic = random.choice(self.improvement_heuristics)
                else:
                    # If we are rebuilding after a massive ruin, use UNIFORM random selection
                    # instead of weighted selection. This prevents the "smart" (greedy) heuristics
                    # from reconstructing the exact same local optimum we just destroyed.
                    if rebuilding_mode:
                        heuristic = random.choice(self.constructive_heuristics)
                    else:
                        heuristic = random.choice(self.constructive_heuristics)
                
                env.run_heuristic(heuristic)
                current_steps += 1
                
                last_value = env.key_value
                if env.key_value > current_best:
                    current_best = env.key_value
                
                # Check if construction just finished
                if env.is_complete_solution:
                    rebuilding_mode = False # Exit rebuilding mode once full
                
            # Phase 2: Improvement (and Perturbation)
            else:
                # Check if we need perturbation
                # Condition 1: Stagnation counter (legacy/safety)
                # Condition 2: All heuristics tried and failed (Immediate Stagnation Detection)
                all_heuristics_failed = (len(tried_heuristics) >= len(self.improvement_heuristics))
                
                if no_improve_steps > max_no_improve or all_heuristics_failed:
                    if all_heuristics_failed:
                        print(f"Run:{run_id} Immediate Stagnation: All {len(tried_heuristics)} improvement heuristics failed. FORCING MASSIVE RUIN.")
                        tried_heuristics.clear() # Reset for next round
                        # Force massive ruin by setting counter above threshold
                        perturbation_count = max_perturbations_before_ruin + 1

                    if self.perturbation_heuristics:
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
                                    
                                heuristic = random.choice(self.perturbation_heuristics)
                                env.run_heuristic(heuristic)
                                removed_count += 1
                                current_steps += 1
                            
                            print(f"  -> Removed {removed_count} nodes. Rebuilding (Randomized Mode)...")
                            perturbation_count = 0
                            no_improve_steps = 0
                            tried_heuristics.clear() # Reset tracking
                            last_value = env.key_value 
                            rebuilding_mode = True # Enable randomized rebuilding
                            continue

                        # Normal (Small) Perturbation
                        # Delete a few nodes (1-5) instead of just 1 to shake it up more
                        perturb_size = random.randint(1, 5)
                        for _ in range(perturb_size):
                            heuristic = random.choice(self.perturbation_heuristics)
                            env.run_heuristic(heuristic)
                        
                        no_improve_steps = 0 # Reset counter
                        tried_heuristics.clear() # Reset tracking
                        last_value = env.key_value
                        # After perturbation, we might be incomplete, so next loop will go to Phase 1
                        continue
                    else:
                        # No perturbation heuristics available, fallback to restart if stuck
                        env.reset()
                        no_improve_steps = 0
                        tried_heuristics.clear() # Reset tracking
                        last_value = 0
                        continue
                
                # Normal Improvement
                if not self.improvement_heuristics:
                     # If no improvement heuristics, just stop or continue random construction (unlikely)
                     break
                
                heuristic = random.choice(self.improvement_heuristics)
                operator = env.run_heuristic(heuristic)
                
                current_steps += 1
                
                # Check if heuristic actually performed an operation (returned a valid operator)
                # If operator is None, it means the heuristic found no valid move (Local Optimum for that heuristic).
                # If operator is valid, the state changed (even if value didn't improve, e.g. side-step).
                is_valid_op = operator is not None and not isinstance(operator, str)
                
                if is_valid_op:
                    # A move was made, so the state has changed.
                    # We reset the 'tried' tracking because previous failures might now be valid in the new state.
                    tried_heuristics.clear()
                    
                    # Check improvement
                    if env.key_value > last_value:
                        last_value = env.key_value
                        no_improve_steps = 0
                        if env.key_value > current_best:
                            current_best = env.key_value
                    else:
                        # Move made but no improvement (Side-step or drop)
                        no_improve_steps += 1
                        last_value = env.key_value
                else:
                    # No move was made (None returned). State is unchanged.
                    # We mark this heuristic as tried for this specific state.
                    tried_heuristics.add(heuristic)
                    no_improve_steps += 1

            # Logging
            current_best = max(current_best, env.key_value)
            if current_steps % 1000 == 0:
                selected_nodes = len(env.current_solution.set_a) + len(env.current_solution.set_b)
                end = datetime.now()
                time_cost = (end - begin).total_seconds()
                print(f"Run:{data}, {experiment}, {run_id}\tsteps:{current_steps}\tselected:{selected_nodes}\ttotal:{node_num}\tnow:{env.key_value}\tcurrent_best:{current_best}\tbest:{env.best_known}\ttime:{time_cost:.2f}", flush=True)

            if env.key_value == env.best_known:
                if env.is_complete_solution and env.is_valid_solution:
                    print(f"!!! NEW BEST FOUND: {env.key_value} > {env.best_known} !!!")
                    env.dump_result(result_file=f"match_best_known_result.txt")
                    found_best = True
                    # Don't stop, try to improve more!
                    env.best_known = env.key_value # Update local best known to keep pushing

            # Check best known
            if env.key_value > env.best_known:
                if env.is_complete_solution and env.is_valid_solution:
                    print(f"!!! NEW BEST FOUND: {env.key_value} > {env.best_known} !!!")
                    env.dump_result(result_file=f"break_best_known_result.txt")
                    found_best = True
                    # Don't stop, try to improve more!
                    env.best_known = env.key_value # Update local best known to keep pushing

        return found_best
