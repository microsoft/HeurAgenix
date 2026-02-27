import os
import random
import math
import pickle
import glob
import uuid
import time
import hashlib
import copy
from datetime import datetime
from src.problems.base.env import BaseEnv
from src.util.util import load_function
from src.problems.max_cut.components import BatchInsertNodeOperator

class PhasedSearchCooperativeHyperHeuristic:
    def __init__(self, heuristic_pool, problem, shared_pool_dir=None, worker_id=None, logger=None):
        self.heuristic_pool_names = heuristic_pool
        self.logger = logger
        self.problem = problem
        self.worker_id = str(worker_id)
        self.shared_pool_dir = shared_pool_dir
        
        # Initialize heuristic lists
        self.constructive_heuristics = []
        self.improvement_heuristics = []
        self.ruin_heuristics = []
        self.breakout_heuristics = {}  # Keep the dictionary for breakout mapping
        
        # Load and Classify Heuristics
        self._classify_heuristics()
        
        self.elite_pool = []
        self.stagnation_level = 0 
        self.consecutive_massive_ruins = 0 
        
        # Distributed Cooperation Setup
        # worker_id is now set in __init__
            
        # Hash worker_id to get a shard index (0-9)
        self.shard_id = int(hashlib.md5(self.worker_id.encode()).hexdigest(), 16) % 10
        
        # Throttling
        self.last_upload_time = 0
        self.last_upload_value = 0
        self.last_sync_time = 0
        self.last_restart_step = -1000 
        
        if self.shared_pool_dir:
            try:
                self._log(f"Shared Elite Pool Directory: {self.shared_pool_dir}")
                os.makedirs(self.shared_pool_dir, exist_ok=True)
            except OSError:
                pass 



    def _classify_heuristics(self):
        # Explicit classifications
        constructive_names = {
            "balance_biased_edge_placement_9f22",
            "balanced_cut_21d5",
            "balanced_cut_c0e6",
            "balanced_random_7f42",
            "heaviest_edge_seed_eb0d",
            "heavy_edge_matching_seed_edd5",
            "highest_delta_node_b31b",
            "highest_delta_edge_9f66",
            "highest_weight_edge_eb0d",
            "highest_weight_edge_eb0c",
            "highest_weight_edge_ca02",
            "most_weight_neighbors_320c",
            "most_weight_neighbors_d31b",
            "random_5c59",
            "semi_greedy_node_grasp_bf9a",
            "softmax_gain_insertion_76de",
            "spectral_seed_fiedler_51e0",
            "continuous_mean_field_batch", 
            "balanced_random_batch", 
            "weighted_degree_batch", 
            "cosm_heuristic_quick",
            "cosm_heuristic_detailed",
        }
        
        improvement_names = {
            "cached_delta_flip_3cfd",
            "first_improvement_flip_7a32",
            "greedy_swap_5bb6",
            "greedy_swap_5bb5",
            "k_block_swap_topk_589e",
            "majority_neighbor_flip_67a0",
            "multi_flip_threshold_fd21",
            "multi_swap_2_dbfe",
            "multi_swap_2_dbff",
            "single_flip_gain_5bb5",
            "tabu_node_flip_cae6",
            "two_node_joint_flip_590a",
        }
        
        # Required Breakout Heuristics
        # We look for these specifically to populate self.breakout_heuristics
        breakout_map = {
            "batch_cluster_ruin": ["batch_cluster_ruin"],
            "batch_worst_ruin": ["batch_worst_ruin"],
            "path_relinking": ["path_relinking_guided_perturbation", "path_relinking"],
            "anti_consensus": ["anti_consensus_perturbation"],
            "batch_flip": ["batch_flip_perturbation"]
        }

        # 1. Classify standard pool
        for h_name in self.heuristic_pool_names:
            base_name = os.path.basename(h_name).replace(".py", "")
            func = load_function(h_name, problem=self.problem)
            
            if base_name in constructive_names:
                self.constructive_heuristics.append(func)
            elif base_name in improvement_names:
                self.improvement_heuristics.append(func)
            
            # Map to breakout dictionary
            for key, variations in breakout_map.items():
                if base_name in variations:
                    self.breakout_heuristics[key] = func

        # 2. Fallback: If breakout heuristics were missed in the main pool, try to load them from standard paths
        # This ensures backward compatibility with the original explicit loading logic.
        base_path = "src/problems/max_cut/heuristics"
        ruin_path = "evolved_heuristics.part3"
        
        fallback_map = {
            "batch_cluster_ruin": os.path.join(base_path, ruin_path, "batch_cluster_ruin.py"),
            "batch_worst_ruin": os.path.join(base_path, ruin_path, "batch_worst_ruin.py"),
            "path_relinking": os.path.join(base_path, "path_relinking_guided_perturbation.py"),
            "anti_consensus": os.path.join(base_path, ruin_path, "anti_consensus_perturbation.py"),
            "batch_flip": os.path.join(base_path, ruin_path, "batch_flip_perturbation.py")
        }
        
        for key, path in fallback_map.items():
            if key not in self.breakout_heuristics:
                try:
                    # Check relative or absolute
                    if os.path.exists(path):
                        self.breakout_heuristics[key] = load_function(path, problem=self.problem)
                    else:
                        abs_path = os.path.join(os.getcwd(), path)
                        if os.path.exists(abs_path):
                            self.breakout_heuristics[key] = load_function(abs_path, problem=self.problem)
                except Exception:
                    pass



    def _get_time_bucket_path(self, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        # YYYYMMDD_HH
        dt = datetime.fromtimestamp(timestamp)
        bucket_name = dt.strftime("%Y%m%d_%H")
        return os.path.join(self.shared_pool_dir, bucket_name)

    def _get_shard_path(self, bucket_path, shard_index):
        return os.path.join(bucket_path, f"shard_{shard_index}")

    def _log(self, message):
        self.logger(message)

    def _save_to_shared_pool(self, item, is_keep_alive=False):
        if not self.shared_pool_dir: return
        
        # [RESEARCH] Direct access, item is always a Wrapper Dict
        solution = item["solution"]
        save_obj = item 
            
        current_time = time.time()
        
        # Throttling Logic (Skip if simply frequent updates of same quality, unless keep-alive)
        if not is_keep_alive:
            if solution.cut_value == self.last_upload_value and (current_time - self.last_upload_time) < 300:
                return # Skip if same value uploaded recently within 5 mins
            
        try:
           
            bucket_path = self._get_time_bucket_path(current_time)
            shard_path = self._get_shard_path(bucket_path, self.shard_id)
            
            # Ensure directories exist (lazy creation)
            if not os.path.exists(shard_path):
                try:
                    os.makedirs(shard_path, exist_ok=True)
                except OSError:
                    pass
            
            # [CRITICAL FIX 2026-02-27] Anti-Homogenization Check (Disk Level)
            # Before writing, check if we already have enough copies of this solution score in the shard.
            # This prevents "Keep-Alive" from flooding the pool with identical solutions (e.g. 5326663.8...).
            # We match the integer part to be fast, then check strict float tolerance relative to filenames if needed.
            # But simply limiting per-integer-bucket is a good heuristic for these large floats.
            
            # Pattern: sol_{val}_...
            # We use a glob to count existing files for this roughly similar score
            score_prefix = f"sol_{int(solution.cut_value)}"
            existing_files = glob.glob(os.path.join(shard_path, f"{score_prefix}*.pkl"))
            
            if len(existing_files) >= 3:
                # Check more strictly: are they actually the same score?
                same_score_count = 0
                target_val = solution.cut_value
                for ef in existing_files:
                    try:
                        # filename format: sol_5326663.8146..._timestamp...
                        fname = os.path.basename(ef)
                        val_str = fname.split("_")[1]
                        val = float(val_str)
                        if abs(val - target_val) < 1e-3:
                            same_score_count += 1
                    except:
                        pass
                
                if same_score_count >= 3:
                    # Too many copies on disk already. Do not save/flood.
                    return

            timestamp_int = int(current_time)
            # Filename: sol_{value}_{timestamp}_{worker}_{rand}.pkl
            filename = f"sol_{solution.cut_value}_{timestamp_int}_{self.worker_id}_{random.randint(1000,9999)}.pkl"
            filepath = os.path.join(shard_path, filename)
            
            # Atomic write
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                pickle.dump(save_obj, f)
            os.rename(temp_path, filepath)
            
            # [LOGGING UPDATE] Print the path of the saved elite/breakthrough
            self._log(f"Saved elite solution to: {filepath}")
            
            # Update throttle stats
            self.last_upload_time = current_time
            self.last_upload_value = solution.cut_value
            
        except Exception as e:
            # Ignore errors (e.g. disk full, permission) to keep running
            pass

    def _sync_shared_pool(self):
        if not self.shared_pool_dir: return
        
        current_time = time.time()
        
        # 1. READ: Scan current hour and previous hour buckets
        buckets_to_scan = []
        
        # Current hour
        current_bucket = self._get_time_bucket_path(current_time)
        buckets_to_scan.append(current_bucket)
        # Previous hour
        prev_bucket = self._get_time_bucket_path(current_time - 3600)
        buckets_to_scan.append(prev_bucket)
        
        # [NEW 2026-02-27] Cross-Hour Migration Strategy
        # If we just crossed into a new hour bucket (e.g. current_bucket is empty or very sparse),
        # we risk "Cold Start Homogenization" where the first few solutions (likely local optima) 
        # dominate the new empty bucket 100%.
        # To prevent this, we forcingly migrate diverse elites from the previous bucket if the new one is empty.
        
        if os.path.exists(prev_bucket) and (not os.path.exists(current_bucket) or len(os.listdir(current_bucket)) < 5):
             # Identify that we are in a transition period.
             # Migration is done distributedly: Each worker checks their own shard.
             prev_shard_path = self._get_shard_path(prev_bucket, self.shard_id)
             curr_shard_path = self._get_shard_path(current_bucket, self.shard_id)
             
             if os.path.exists(prev_shard_path):
                 try:
                     # Create current shard if needed
                     os.makedirs(curr_shard_path, exist_ok=True)
                     
                     # Read TOP 30 UNIQUE solutions from previous shard
                     # Use strict 1e-3 difference to ensure diversity
                     files = glob.glob(os.path.join(prev_shard_path, "*.pkl"))
                     
                     # 1. Parse all files and store as (score, filepath)
                     candidates = []
                     for f in files:
                         try:
                             # filename format: sol_5326663.8146..._timestamp...
                             fname = os.path.basename(f)
                             val_str = fname.split("_")[1]
                             val = float(val_str)
                             candidates.append((val, f))
                         except: pass
                     
                     # 2. Sort by Score Descending (Quality First)
                     candidates.sort(key=lambda x: x[0], reverse=True)
                     
                     # 3. Select unique solutions (Difference > 1e-3)
                     # We only keep the FIRST occurrence of any score (highest quality duplicate if any)
                     files_to_migrate = []
                     selected_scores = []
                     
                     for score, fpath in candidates:
                         is_duplicate = False
                         for existing_score in selected_scores:
                             if abs(score - existing_score) < 1e-3:
                                 is_duplicate = True
                                 break
                         
                         if not is_duplicate:
                             files_to_migrate.append(fpath)
                             selected_scores.append(score)
                             
                         if len(files_to_migrate) >= 30:
                             break
                     
                     # Copy them to new bucket
                     
                     # Copy them to new bucket
                     import shutil
                     for old_f in files_to_migrate:
                         new_f = os.path.join(curr_shard_path, os.path.basename(old_f))
                         if not os.path.exists(new_f):
                             shutil.copy2(old_f, new_f)
                             
                     if files_to_migrate:
                         self._log(f"Migrated {len(files_to_migrate)} diverse elites from {os.path.basename(prev_bucket)} to {os.path.basename(current_bucket)}")
                         
                 except Exception as e:
                     # self._log(f"Migration error: {e}")
                     pass

        files_to_read = []
        
        for bucket in buckets_to_scan:
            if not os.path.exists(bucket): continue
            
            # Randomly pick 2-3 shards to check in this bucket (Statistically sufficient)
            # We assume shards 0-9 exist
            shards = random.sample(range(10), 3) 
            
            for s_idx in shards:
                s_path = self._get_shard_path(bucket, s_idx)
                if os.path.exists(s_path):
                    try:
                        # Glob only one shard, much faster
                        shard_files = glob.glob(os.path.join(s_path, "*.pkl"))
                        # Take random sample if too many in one shard
                        if len(shard_files) > 20: 
                            shard_files = random.sample(shard_files, 20)
                        files_to_read.extend(shard_files)
                    except:
                        pass
        
        # Process files
        for fpath in files_to_read:
            try:
                with open(fpath, 'rb') as f:
                    data = pickle.load(f)
                    # Compatibility: If loaded data is Solution object, wrap it
                    if not isinstance(data, dict):
                         data = {"solution": data, "history": []}
                    self._add_to_local_pool(data, share=False)
            except:
                pass
                
        # 2. WRITE (Keep-Alive): Periodically re-upload top elites to current bucket
        # Do this roughly every 5th sync (e.g. every ~250 steps)
        # Using simple random check 20%
        if random.random() < 0.2 and self.elite_pool:
            # Pick top 3 from local pool
            # Sort local pool by value descending
            # [COMPATIBILITY] Handle Wrapper
            sorted_pool = sorted(self.elite_pool, key=lambda s: s["solution"].cut_value, reverse=True)
            top_elites = sorted_pool[:3]
            
            for elite in top_elites:
                self._save_to_shared_pool(elite, is_keep_alive=True)

    def _add_to_local_pool(self, item, share=True):
        # [RESEARCH] Direct access, item is always a Wrapper Dict
        # Backward compatibility: If item is solution object, wrap it
        if not isinstance(item, dict):
             item = {"solution": item, "history": []}
             
        solution_ref = item["solution"]
        history_ref = list(item.get("history", []))
        
        # [DYNAMIC TABU STRATEGY 2026-02-19]
        if not hasattr(self, "visited_peaks"):
             self.visited_peaks = {}
             
        TABU_TOLERANCE = 1e-3
        MAX_VISITS_PER_PEAK = 20
        
        for peak_val, count in self.visited_peaks.items():
            if count >= MAX_VISITS_PER_PEAK and abs(solution_ref.cut_value - peak_val) < TABU_TOLERANCE:
                return

        # Add copy of solution to pool
        from src.problems.max_cut.components import Solution
        
        # Deep copy the sets for storage
        new_sol = Solution(set(solution_ref.set_a), set(solution_ref.set_b), solution_ref.cut_value)
        # Create new wrapper
        new_wrapper = {"solution": new_sol, "history": history_ref}
        
        # [IMPROVED DIVERSITY CONTROL]
        def calc_dist_pool(s1, s2):
            d1 = len((s1.set_a & s2.set_b) | (s1.set_b & s2.set_a))
            d2 = len((s1.set_a & s2.set_a) | (s1.set_b & s2.set_b))
            return min(d1, d2)

        node_num = len(new_sol.set_a) + len(new_sol.set_b)
        
        # ------------------------------------------------------------------
        # Strategy 1: Score-based Duplication Check (New Logic)
        # ------------------------------------------------------------------
        # [MODIFIED 2026-02-26] Allow up to 3 solutions with the same score (User Request)
        # However, use a wider tolerance (1e-3) to treat "jittered" values as identical.
        # This prevents the pool from filling with 20 copies of 5317.000001, 5317.000002, etc.
        same_score_count = 0
        pool_tolerance = 1e-3
        
        for existing_wrapper in self.elite_pool:
            if abs(new_sol.cut_value - existing_wrapper["solution"].cut_value) < pool_tolerance:
                same_score_count += 1
        
        if same_score_count >= 3:
            # We already have enough (3) representatives of this score range. Reject.
            return

        # Policy B: Distinct Solution (Add/Evict)
        
        # If pool is not full, just add it.
        if len(self.elite_pool) < 20:
            self.elite_pool.append(new_wrapper)
            if share: self._save_to_shared_pool(new_wrapper)
            return

        # Find worse solution (using wrappers)
        
        pool_values = []
        for e in self.elite_pool:
            idx_val = e["solution"].cut_value
            pool_values.append(idx_val)
            
        min_val = min(pool_values)
        
        # Case 1 & 2: Better or Equal to worst
        if new_sol.cut_value >= min_val:
            # Replace the worst one
            # Note: Since we passed the "Score Check" above, we know we aren't flooding.
            
            for i, s in enumerate(self.elite_pool):
                if s["solution"].cut_value == min_val:
                    self.elite_pool[i] = new_wrapper
                    if share: self._save_to_shared_pool(new_wrapper)
                    break
            return

        # Case 3: Worse than worst -> REJECT
        # [MODIFIED 2026-02-26] Removed "Stranger Admission" logic.
        # We value quality first. If it's worse than the worst elite, it's out.
        else:
             return

    # Removed _calculate_consensus_flip (logic moved to anti_consensus_perturbation heuristic)


    def _update_elite_pool_from_env(self, env):
        # [NEW 2026-02-26] Capture solution + history for full reproducibility
        sol = env.current_solution
        # Deep copy history to ensure it's frozen at this point
        history = list(env.recordings) if env.recordings else []
        package = {"solution": sol, "history": history}
        self._add_to_local_pool(package, share=True)
    
    def _run_improvement_phase(self, env):
        # [VND Implementation with Strict Hill Climbing]
        # Iterate through all available heuristics until no improvement is found.
        # This converts "relaxed" search into "intensive polishing".
        if not self.improvement_heuristics: return False
        
        # Limit max VND iterations to prevent infinite loops (though strict ascent prevents cycling, costs time)
        # CHANGED: Increased from 10 to 10000000. 
        # Since strict ascent is enforced (env.key_value > start_val), infinite loops are impossible
        # unless the score increases indefinitely, which is impossible for MaxCut.
        # Convergence is guaranteed. We want FULL convergence.
        max_vnd_loops = 10000000 
        total_improved = False
        
        # Pre-shuffle heuristics to improve robustness
        # We work on a copy of the list to shuffle it
        heuristics_queue = list(self.improvement_heuristics)
        
        for loop_idx in range(max_vnd_loops):
            improved_in_this_loop = False
            random.shuffle(heuristics_queue)
            
            for heuristic in heuristics_queue:
                # 1. Snapshot State
                # Deep copy is needed for components.Solution
                from src.problems.max_cut.components import Solution
                backup_sol = Solution(set(env.current_solution.set_a), 
                                      set(env.current_solution.set_b), 
                                      env.current_solution.cut_value)
                start_val = backup_sol.cut_value
                # Backup recordings length to rollback changes if heuristic fails
                recordings_len = len(env.recordings) if env.recordings else 0
                
                # 2. Run Heuristic (In-Place Modification)
                try:
                    env.run_heuristic(heuristic)
                except Exception as e:
                    # Sparse logging to prevent explosion if heuristic is fundamentally broken
                    if not hasattr(self, "_error_log_count"): self._error_log_count = 0
                    self._error_log_count += 1
                    if self._error_log_count < 10 or self._error_log_count % 1000 == 0:
                         self._log(f"Error running heuristic {heuristic.__name__}: {e}")
                    
                    env.current_solution = backup_sol
                    # Rollback recordings on error
                    if env.recordings:
                        env.recordings = env.recordings[:recordings_len]
                    continue

                # 3. Acceptance Criteria: Strict Ascent
                # If Score Dropped or Equal -> Revert (We want to find peaks, not drift)
                if env.key_value <= start_val:
                    # Revert
                    env.current_solution = backup_sol
                    # Restore env properties just in case
                    env.current_solution.cut_value = start_val
                    env.problem_state = env.get_problem_state() 
                    # Rollback recordings for rejected move
                    if env.recordings:
                        env.recordings = env.recordings[:recordings_len]
                    
                    # Note: We assume env.problem_state is derived from current_solution, 
                    # but heuristic might modify algorithm_data too. Usually negligible for basic heuristics.
                else:
                    # Accepted
                    improved_in_this_loop = True
                    total_improved = True
                    # Optimization: If we found a gain, we might want to stick with this heuristic or continue?
                    # Standard VND continues to next heuristic.
            
            # If a full pass through all heuristics yielded no gain, we are at a local optimum for ALL neighborhoods.
            if not improved_in_this_loop:
                break
                
        return total_improved

    def _apply_breakout(self, env, strategy):
        node_num = env.instance_data["node_num"]
        
        if strategy == "supernova_ruin":
            # "Anti-Consensus" Strategy: Flip stable variables
            # Increase intensity to 10%-20% to escape deep basin
            ratio = random.uniform(0.10, 0.20)
            
            # Prepare algorithm context for heuristic (unpack Elite Pool wrappers)
            if self.elite_pool:
                env.algorithm_data["elite_pool"] = [s["solution"] for s in self.elite_pool]
            
            # Use evolved heuristic "anti_consensus"
            if "anti_consensus" in self.breakout_heuristics:
                 h = self.breakout_heuristics["anti_consensus"]
                 env.run_heuristic(h, parameters={"ratio": ratio})
                 self._log(f"Supernova Ruin applied: Anti-Consensus Flip (Ratio: {ratio:.2f}).")
            else:
                 # Fallback if heuristic missing (should be loaded by default)
                 self._log("Supernova Ruin: Heuristic missing, falling back to Heavy Ruin.")
                 self._apply_breakout(env, "heavy_ruin")

        elif strategy == "active_pool_relinking":
            # [NEW] Active strategy: Force path relinking between distant elites
            if len(self.elite_pool) < 2:
                 return
            
            # 1. Find Best Known (Handle Wrapper)
            best_wrapper = max(self.elite_pool, key=lambda s: s["solution"].cut_value)
            best_sol = best_wrapper["solution"]
            
            # 2. Find a "Distant" High-Quality Elite
            # [FIX] Lower threshold to 0.97 to ensure our current elites (2400-2406) can participate
            # 2446 * 0.97 = 2372, so 2400+ are valid candidates
            candidates = [s for s in self.elite_pool if s["solution"].cut_value > env.best_known * 0.97]
            if not candidates: 
                 return
                 
            # Helper for distance
            def calc_dist(s1, s2):
                d1 = len((s1.set_a & s2.set_b) | (s1.set_b & s2.set_a))
                d2 = len((s1.set_a & s2.set_a) | (s1.set_b & s2.set_b))
                return min(d1, d2)
            
            # Find candidate farthest from best_sol (Need to unwrap candidate for dist calculation)
            # [FIX] Use weighted random choice to avoid "Groundhog Day" repeating the same link
            # Pick from top 3 farthest
            candidates.sort(key=lambda s: calc_dist(s["solution"], best_sol), reverse=True)
            top_candidates = candidates[:min(3, len(candidates))]
            distant_wrapper = random.choice(top_candidates)
            distant_elite = distant_wrapper["solution"]
            
            dist = calc_dist(best_sol, distant_elite)
            
            # [IMPROVED 2026-02-18] Dynamic Threshold based on Graph Size
            # Hardcoding 'dist < 40' or '200' is bad because it ignores problem scale.
            # Use 1.5% of total nodes as the "too close" threshold.
            
            # CRITICAL FIX: node_num is not in self.problem, it is in env.instance_data
            node_num = env.instance_data["node_num"]
            # [FIX 2026-02-19] Increased safety radius to 2.5% to prevent black hole collapse
            threshold = max(10, int(node_num * 0.025))

            if dist < threshold: 
                 # [FIX 2026-02-19] Improved Robustness:
                 # If targets are too close, standard Path Relinking is weak.
                 # Instead of skipping or punishing, we force a "Micro-Perturbation" to break strict convergence.
                 # This helps exploring the immediate neighborhood of the basin.
                 self._log(f"Active Relinking: Targets too close (Dist={dist} < Threshold={threshold}). Triggering Micro-Perturbation.")
                 
                 # Load best solution (WITH HISTORY)
                 env.current_solution = copy.deepcopy(best_sol)
                 env.recordings = list(best_wrapper.get("history", []))

                 env.current_solution.cut_value = best_sol.cut_value
                 
                 # Perturb 2% of nodes (enough to move away ~60 nodes in 3000)
                 # This is lighter than Level 1 Stagnation (Light Ruin), keeping us in the same "Peak Family".
                 if "batch_flip" in self.breakout_heuristics:
                     h = self.breakout_heuristics["batch_flip"]
                     env.run_heuristic(h, parameters={"ratio": 0.02})
                 else:
                     micro_flip_count = max(5, int(node_num * 0.02))
                     nodes_to_flip = random.sample(range(node_num), micro_flip_count)
                     op = BatchInsertNodeOperator(
                         [n for n in nodes_to_flip if n in env.current_solution.set_b],
                         [n for n in nodes_to_flip if n in env.current_solution.set_a]
                     )
                     env.run_operator(op)
                 
                 return

            self._log(f"*** ACTIVE RELINKING: Best({best_sol.cut_value}) <-> Distant({distant_elite.cut_value}, Dist={dist}) ***")

            # 3. Reset to Best, Target = Distant (WITH HISTORY)
            env.current_solution = copy.deepcopy(best_sol)
            env.recordings = list(best_wrapper.get("history", []))
            env.current_solution.cut_value = best_sol.cut_value
            
            # Pass unwrapped distant elite
            env.algorithm_data["elite_pool"] = [distant_elite] 
                
            if "path_relinking" in self.breakout_heuristics:
                h = self.breakout_heuristics["path_relinking"]
                # Move 40% towards the other peak
                env.run_heuristic(h, parameters={"intensity": 0.4})
                
                # [FIX]: Immediate Local Optimization in the Valley
                if self.improvement_heuristics:
                     self._log("Rapid Mining in Valley...")
                     # Execute 2 rounds of improvement to settle into a local optimum
                     self._run_improvement_phase(env)
                     self._run_improvement_phase(env)

        elif strategy == "path_relinking_to_best" and "path_relinking" in self.breakout_heuristics:
             # Targeted PR: Force link towards the absolute Best Known in the pool
             h = self.breakout_heuristics["path_relinking"]
             # Filter pool to only include the BEST solution(s) to guarantee target direction
             best_val = max(s["solution"].cut_value for s in self.elite_pool)
             # Unwrap solutions for heuristics
             best_solutions = [s["solution"] for s in self.elite_pool if s["solution"].cut_value == best_val]
             
             # Create a focused context
             env.algorithm_data["elite_pool"] = best_solutions
             # Run with high intensity
             env.run_heuristic(h, parameters={"intensity": 1.0})
             # Restore full pool for other operations (though reference is passed, we shouldn't damage self.elite_pool)
             # NOTE: Since we pass list by ref, safest is to NOT modify self.elite_pool. 
             # But here we temporarily overwrote algorithm_data entry, which is fine.
             self._log(f"Targeted Path Relinking -> Best Known ({best_val})")

             # [FIX] Dig deeper around the path
             if self.improvement_heuristics:
                 self._log("Mining Path to Best...")
                 self._run_improvement_phase(env)
                 self._run_improvement_phase(env)

        elif strategy == "path_relinking" and "path_relinking" in self.breakout_heuristics:
            # Parameters: intensity
            h = self.breakout_heuristics["path_relinking"]
            # Pass elite_pool via algorithm_data
            # We need to hack/inject elite_pool into algorithm_data if not present
            # env.run_heuristic passes env.problem_state and env.algorithm_data
            env.algorithm_data["elite_pool"] = [s["solution"] for s in self.elite_pool]
            env.run_heuristic(h, parameters={"intensity": 0.3})
            
        elif strategy == "jump_to_secondary_peak":
             # Strategy: Teleport to a high-quality local optimum
             if not self.elite_pool:
                 self._apply_breakout(env, "supernova_ruin")
                 return
             
             best_val = max(s["solution"].cut_value for s in self.elite_pool)
             # Candidates: High quality but strictly less than Best Known (to find secondary peaks)
             # We want to revisit peaks like 26992 to see if we can sharpen them
             # [FIX] Use relative threshold for large-weight instances (imgseg)
             threshold = max(150, env.best_known * 0.02)
             
             # Filter wrappers/items based on unwrapped value
             candidates = [s for s in self.elite_pool if s["solution"].cut_value >= env.best_known - threshold and s["solution"].cut_value < best_val]
             
             desc = "SECONDARY PEAK"

             # [FIX] If no secondary peak, jump to a DISTANT parallel peak (same best value)
             if not candidates:
                 def calc_dist_j(s1, s2):
                    d1 = len((s1.set_a & s2.set_b) | (s1.set_b & s2.set_a))
                    d2 = len((s1.set_a & s2.set_a) | (s1.set_b & s2.set_b))
                    return min(d1, d2)
                 
                 # Look for solutions with SAME best value but Distance > 400
                 current_sol = env.current_solution
                 # Handle wrapper
                 candidates = [s for s in self.elite_pool if abs(s["solution"].cut_value - best_val) <= 1e-3 and calc_dist_j(s["solution"], current_sol) > 400]
                 desc = "PARALLEL UNIVERSE PEAK"
             
             if candidates:
                 target_item = random.choice(candidates)
                 target_sol = target_item["solution"]
                 
                 # Deep copy
                 from src.problems.max_cut.components import Solution
                 new_sol = Solution(set(target_sol.set_a), set(target_sol.set_b), target_sol.cut_value)
                 env.current_solution = new_sol
                 
                 # Restore History
                 env.recordings = list(target_item.get("history", []))
                     
                 # Verify value
                 env.current_solution.cut_value = env.get_key_value(env.current_solution)
                 # Sync problem state
                 env.problem_state = env.get_problem_state()
                 self._log(f"*** JUMPED TO {desc}: {env.current_solution.cut_value} (from pool of {len(candidates)}) ***")
             else:
                 # If no secondary peak found, try Supernova
                 self._apply_breakout(env, "supernova_ruin")

        elif strategy == "light_ruin":
            # Cluster Ruin 1%
            if "batch_cluster_ruin" in self.breakout_heuristics:
                h = self.breakout_heuristics["batch_cluster_ruin"]
                count = max(10, int(node_num * 0.01))
                env.run_heuristic(h, parameters={"count": count})
            else:
                # Fallback
                self._apply_breakout(env, "medium_ruin")
                
        elif strategy == "medium_ruin":
             # Worst Ruin 5%
            if "batch_worst_ruin" in self.breakout_heuristics:
                h = self.breakout_heuristics["batch_worst_ruin"]
                count = max(50, int(node_num * 0.05))
                env.run_heuristic(h, parameters={"count": count})
        
        elif strategy == "heavy_ruin":
            # Cluster Ruin 10-20%
            if "batch_cluster_ruin" in self.breakout_heuristics:
                h = self.breakout_heuristics["batch_cluster_ruin"]
                count = max(200, int(node_num * 0.15))
                env.run_heuristic(h, parameters={"count": count})

        elif strategy == "massive_ruin":
            # [OPTIMIZED 2026-02-16]
            # Strategic Reconstructive Ruin (20%-50% Destruction + Cosm Repair)
            # This balances Exploration (jumping out of basin) and Exploitation (using learned structure).
            
            if "batch_worst_ruin" in self.breakout_heuristics:
                h_ruin = self.breakout_heuristics["batch_worst_ruin"]
                
                # Dynamic Ratio: 20% to 50%
                # Lower bound (20%) allows "Large Step" optimization
                # Upper bound (50%) allows "Basin Hopping"
                ratio = random.uniform(0.20, 0.50)
                count = int(node_num * ratio)
                
                self._log(f"RECONSTRUCTIVE RUIN: Removing {count} nodes ({ratio:.1%}) to trigger repair...")
                env.run_heuristic(h_ruin, parameters={"count": count})
                
                # [Repair Phase]
                # Essential: Use Cosm to fill the holes intelligently
                if self.constructive_heuristics:
                    repair_h = [h for h in self.constructive_heuristics if "cosm" in h.__name__ and "detailed" in h.__name__]
                    if not repair_h:
                        repair_h = [h for h in self.constructive_heuristics if "cosm" in h.__name__]
                    
                    if repair_h:
                         # self._log(f"Repairing with {repair_h[0].__name__}...")
                         # Cosm checks solution state and fills unselected_nodes
                         env.run_heuristic(repair_h[0])
                
                # [MODIFIED 2026-02-18] Add Noise Injection to prevent Loop
                # Even after ruin, COSM might reconstruct the exact same solution.
                # We force a small random perturbation (5%) to ensure we land in a NEW basin.
                ratio_noise = 0.05
                if "batch_flip" in self.breakout_heuristics:
                     h = self.breakout_heuristics["batch_flip"]
                     env.run_heuristic(h, parameters={"ratio": ratio_noise})
                     self._log(f"Noise Injection: Random Flip ({ratio_noise:.1%}) to escape basin.")
                else:
                     noise_nodes = random.sample(range(node_num), int(node_num * ratio_noise))
                     op_noise = BatchInsertNodeOperator(
                         [n for n in noise_nodes if n in env.current_solution.set_b],
                         [n for n in noise_nodes if n in env.current_solution.set_a]
                     )
                     env.run_operator(op_noise)
                     self._log(f"Noise Injection: Flipped {len(noise_nodes)} nodes ({ratio_noise:.1%}) to escape basin.")
                
                # Force update value
                cur_val = env.get_key_value(env.current_solution)
                if env.current_solution.cut_value != cur_val:
                    env.current_solution.cut_value = cur_val

            else:
                 # Fallback: Random Flip 40%
                 if "batch_flip" in self.breakout_heuristics:
                     h = self.breakout_heuristics["batch_flip"]
                     env.run_heuristic(h, parameters={"ratio": 0.40})
                     self._log("Fallback Ruin: Random Flip (40%).")
                 else:
                     nodes = random.sample(range(node_num), int(node_num * 0.40))
                     op = BatchInsertNodeOperator(
                         [n for n in nodes if n in env.current_solution.set_b],
                         [n for n in nodes if n in env.current_solution.set_a],
                     )
                     env.run_operator(op)
                     self._log(f"Fallback Ruin: Random Flipped {len(nodes)} nodes.")

        elif strategy == "soft_restart":
             self._log("... Soft Restart Triggered ... Abandoning current solution.")
             
             # Option A: Jump to a random Elite (preferably one we haven't visited lately)
             force_constructive = False
             
             if self.elite_pool and len(self.elite_pool) > 5:
                  # Pick a random elite, but favor those DIFFERENT from current
                  # Calculate distance to current
                  def calc_dist_r(s1, s2):
                        d1 = len((s1.set_a & s2.set_b) | (s1.set_b & s2.set_a))
                        d2 = len((s1.set_a & s2.set_a) | (s1.set_b & s2.set_b))
                        return min(d1, d2)
                  
                  # Sort by distance descending (furthest first)
                  sorted_elites = sorted(self.elite_pool, key=lambda s: calc_dist_r(s["solution"], env.current_solution), reverse=True)
                  # Pick from top 5 furthest
                  target_item = random.choice(sorted_elites[:5])
                  target_sol = target_item["solution"]
                  
                  # [CRITICAL FIX 2026-02-19] Check if the "furthest" elite is actually distant.
                  # If the pool has collapsed (Homogenized), the furthest elite might be just 10 flips away.
                  # In that case, a Soft Restart is useless. We must force a HARD RESTART (Constructive).
                  dist = calc_dist_r(target_sol, env.current_solution)
                  min_restart_dist = max(50, int(node_num * 0.05)) # e.g. 150 nodes for 3000 node graph
                  
                  if dist < min_restart_dist:
                       self._log(f"Soft Restart Aborted: Pool Homogenized (Max Dist={dist} < {min_restart_dist}). Forcing Hard Constructive Restart.")
                       force_constructive = True
                  else:
                       from src.problems.max_cut.components import Solution
                       env.current_solution = Solution(set(target_sol.set_a), set(target_sol.set_b), target_sol.cut_value)
                       # Restore History
                       env.recordings = list(target_item.get("history", []))

                       self._log(f"Restarted from Distant Elite (Val: {target_sol.cut_value}, Dist: {dist})")
             
             else:
                  force_constructive = True
                  
             if force_constructive:
                  # Option B: Complete Noise Restart (if pool is empty or small OR homogenized)
                  # Or Constructive Restart
                  self._log("Restarting with Constructive Heuristic (High Quality)...")
                  env.reset(output_dir=env.output_dir)
                  
                  # [SYNC WITH COLD START] Use best constructive heuristics to reach High Basin
                  construction_steps = 0
                  while not env.is_complete_solution and construction_steps < 1000:
                      if not self.constructive_heuristics:
                          break
                      
                      # Strict Priority: Detailed > Quick > Mean Field
                      if construction_steps == 0:
                           detailed_cosm = [h for h in self.constructive_heuristics if "cosm_heuristic_detailed" in h.__name__]
                           other_cosm = [h for h in self.constructive_heuristics if ("cosm" in h.__name__ or "mean_field" in h.__name__) and "detailed" not in h.__name__]
                           
                           if detailed_cosm:
                               h = detailed_cosm[0] # Always pick detailed if available
                           elif other_cosm:
                               h = random.choice(other_cosm)
                           else:
                               h = random.choice(self.constructive_heuristics)
                      else:
                          h = random.choice(self.constructive_heuristics)
                          
                      env.run_heuristic(h)
                      construction_steps += 1
                  
                  # [FIX 2026-02-19] Set Immunity Timer
                  self.last_restart_step = self.current_run_steps
                  self._log(f"Immunity Activated for 500 steps (Restart Step: {self.current_run_steps})")


             # Sync state
             env.problem_state = env.get_problem_state()
             env.current_solution.cut_value = env.get_key_value(env.current_solution)


    def run(self, env: BaseEnv) -> bool:
        # [REFACTORED for Cooperative Search - Cold Start Only]
        
        # Explicitly maximize chances by syncing first (populate pool for interactions later)
        self._sync_shared_pool()

        self._log("Switching to Constructive Phase (Cold Start)...")
        # Fallback: Construct New Solution if no Best Known file
        # Loop until solution is COMPLETE and VALID
        max_retries = 10
        for retry in range(max_retries):
            
            # Reset environment for a fresh start
            env.reset(output_dir=env.output_dir)
            
            # Keep constructing until complete
            construction_steps = 0
            while not env.is_complete_solution and construction_steps < 1000:
                if not self.constructive_heuristics:
                    break
                
                # Prefer Cosm for first attempt as it is SOTA for these graphs
                if retry == 0 and construction_steps == 0:
                        # Strict Priority: Detailed > Quick > Mean Field
                        detailed_cosm = [h for h in self.constructive_heuristics if "cosm_heuristic_detailed" in h.__name__]
                        other_cosm = [h for h in self.constructive_heuristics if ("cosm" in h.__name__ or "mean_field" in h.__name__) and "detailed" not in h.__name__]
                        
                        if detailed_cosm:
                            h = detailed_cosm[0] # Always pick detailed if available
                        elif other_cosm:
                            h = random.choice(other_cosm)
                        else:
                            h = random.choice(self.constructive_heuristics)
                else:
                    h = random.choice(self.constructive_heuristics)
                    
                env.run_heuristic(h)
                construction_steps += 1
            
            if env.is_complete_solution and env.key_value > 100:
                self._log(f"Construction completed. Value: {env.key_value}")
                break
            else:
                self._log(f"Construction failed or incomplete (Value: {env.key_value}). Retrying ({retry+1}/{max_retries})...")
        
        if not env.is_complete_solution:
                self._log("Critical Failure: Unable to construct valid solution after retries.")
                return False
            
        current_best = env.key_value
        self._update_elite_pool_from_env(env)
        
        no_improve_steps = 0

        # CHANGE: Use instance variable to track steps for coordination with restart logic
        self.current_run_steps = 0
        
        self._log(f"Starting Breakout Search from {current_best}...")

        # [DYNAMIC TABU LIST]
        # Keep track of local optima we have visited frequently.
        # Format: {cut_value: visit_count}
        self.visited_peaks = {}
        # [FIX: RELAX TOLERANCE]
        # Strict 1e-6 is too tight for float variations on different heuristic paths.
        # Use 1e-3 (or even 0.1) since object function values are usually large integers/floats.
        # For MaxCut with float weights, peaks are usually separated by significant margins.
        TABU_TOLERANCE = 1e-3 
        MAX_VISITS_PER_PEAK = 20 # How many times can we rediscover the same peak before banning it?

        while env.continue_run:
            self.current_run_steps += 1
            
            # [DYNAMIC ANTI-GRAVITY SHIELD]
            # Instead of hardcoding, we check if the current value has been "exhausted".
            current_val = env.key_value
            is_tabu = False
            
            # Check if we are in a Forbidden Peak
            for peak_val, count in self.visited_peaks.items():
                if count >= MAX_VISITS_PER_PEAK and abs(current_val - peak_val) < TABU_TOLERANCE:
                     is_tabu = True
                     # Only log sparingly
                     if self.current_run_steps % 100 == 0:
                         self._log(f"In Exhausted Basin ({peak_val}). Triggering Evacuation.")
                     break
            
            if is_tabu:
                # Force massive ruin (Supernova) to escape processing this dead zone
                # [FIX 2026-02-27] Preventing infinite loop if heuristic fails to change solution
                prev_tabu_val = env.key_value
                self._apply_breakout(env, "supernova_ruin")
                
                # Check if we actually moved out of the basin
                if abs(env.key_value - prev_tabu_val) < 1e-3:
                     self._log("Supernova failed to break Tabu (Value unchanged). Forcing Random Ruin.")
                     # Fallback to pure random ruin which is guaranteed to change state
                     # Use batch_flip explicitly if available, otherwise massive_ruin which has fallbacks
                     if "batch_flip" in self.breakout_heuristics:
                         h = self.breakout_heuristics["batch_flip"]
                         # Flip 20%
                         env.run_heuristic(h, parameters={"ratio": 0.20})
                     else:
                         self._apply_breakout(env, "massive_ruin")
                     
                # [SAFETY BREAK 2026-02-27] Final check to prevent ANY infinite loop
                if abs(env.key_value - prev_tabu_val) < 1e-3:
                     self._log("CRITICAL: Breakout failed to change solution. Forcing escape from Tabu block.")
                     # If we can't move, we must let the main loop proceed, 
                     # even if it means researching the same peak (which will trigger standard stagnation logic).
                     # We might be at a global optimum where no move is possible? (Unlikely for MaxCut)
                     pass
                else:
                    # Successful move, loop back to start to re-evaluate new position
                    continue

            # --- Phase A: Repair / Improve ---
            # Try to improve current solution (which might be ruined)
            improved = self._run_improvement_phase(env)
            
            # [CRITICAL FIX 2026-02-26] Detect Stagnation at Local Optima
            # If we are at a local optimum (not improving), we must record this peak
            # to prevent infinite cycling around the same basin.
            if not improved:
                 peak_val = env.key_value
                 found_family = False
                 for existing_val in list(self.visited_peaks.keys()):
                     if abs(peak_val - existing_val) < TABU_TOLERANCE:
                         self.visited_peaks[existing_val] += 1
                         found_family = True
                         break
                 if not found_family:
                     self.visited_peaks[peak_val] = 1
            
            # --- Phase B: Check Status ---
            # 1. Update Global/Local Best
            if env.key_value > current_best:
                current_best = env.key_value
                
                # [DYNAMIC TABU LIST UPDATE]
                # We found a new peak. Record it.
                # Linear scan to see if it belongs to an existing family
                found_family = False
                for existing_val in list(self.visited_peaks.keys()): # List copy as we might modify
                    if abs(current_best - existing_val) < TABU_TOLERANCE:
                        # Update the peak definition to the better value
                        count = self.visited_peaks[existing_val]
                        del self.visited_peaks[existing_val]
                        self.visited_peaks[current_best] = count + 1
                        found_family = True
                        break
                
                if not found_family:
                    # New distinct peak
                    self.visited_peaks[current_best] = 1
                
                no_improve_steps = 0
                self.stagnation_level = 0
                self.consecutive_massive_ruins = 0 # Reset panic counter on improvement
                self._update_elite_pool_from_env(env)
                self._log(f"Step:{self.current_run_steps} NEW LOCAL BEST: {current_best}")
                

                if current_best > env.best_known:
                    self._log(f"!!! BREAKTHROUGH: {current_best} > {env.best_known} !!!")
                    env.best_known = current_best
                    
                    # [SAFE SAVE STRATEGY] Only save if strictly better than anything on disk
                    saved_best = 0.0
                    if os.path.exists(env.output_dir):
                        for f in os.listdir(env.output_dir):
                            if f.startswith("breakthrough_") or f.startswith("match_"):
                                try:
                                    # Format: breakthrough_..._SCORE.txt
                                    part = f.rsplit("_", 1)[-1] 
                                    score = float(part.replace(".txt", ""))
                                    if score > saved_best:
                                        saved_best = score
                                except: pass
                    
                    if (current_best - saved_best) > 1e-3:
                        env.dump_result(result_file=f"breakthrough_from_worker_{self.worker_id}_{current_best}.txt")

                elif abs(current_best - env.best_known) < 1e-3:
                     self._log(f"~~~ MATCHED BEST KNOWN: {current_best} ~~~")
                     # Only save Match if no results exist yet
                     has_records = False
                     if os.path.exists(env.output_dir):
                         for f in os.listdir(env.output_dir):
                             if f.startswith("breakthrough_") or f.startswith("match_"):
                                 has_records = True
                                 break
                     
                     if not has_records:
                        env.dump_result(result_file=f"match_best_known_from_worker_{self.worker_id}_{current_best}.txt")
            else:
                no_improve_steps += 1
                
                # 2. Relaxed Elite Pool Update (Fix for Path Relinking)
                # If we are stuck but the solution is still decent (e.g. > 98% of BK)
                # we add it to the pool to provide diversity for path relinking.
                # Don't add every step, maybe every 10 steps to avoid flooding with identical copies
                is_best_known = env.key_value >= env.best_known
                if is_best_known or (env.key_value >= env.best_known * 0.98 and self.current_run_steps % 10 == 0):
                    self._update_elite_pool_from_env(env)

            # --- Phase C: Breakout / Ruin Strategies ---
            # Adaptive Patience based on problem size
            # Large instances (e.g. 100k nodes) need significant time to explore deep basins.
            node_num = env.instance_data.get("node_num", 1000)
            base_patience = max(500, int(node_num / 10)) # e.g. 100k nodes -> 10,000 steps check interval? No, keep it responsive.
            # Updated 2026-02-24: Use a balanced patience. 
            # For 100k nodes, we check every ~2000 steps.
            patience = min(2000, max(200, int(node_num / 50)))
            
            if no_improve_steps > patience:
                # Instead of immediate level jump, we use a "Retry Budget" based on graph size.
                # Complex graphs need more retries at each intensity level before giving up.
                
                # Dynamic Retry Thresholds calculated from node_num
                # Small graph (800): ~4 retries. Large graph (100k): ~20 retries allowed per phase.
                # Logic: Don't give up strictly on a strategy until we've tried it enough times relative to problem complexity.
                max_retries_per_phase = max(3, int(math.log10(node_num) * 2)) 
                
                # Increment internal counter for current phase
                if not hasattr(self, 'phase_retries'):
                    self.phase_retries = 0
                
                self.phase_retries += 1
                
                # Map simple 4 levels (1, 2, 3, 4) based on how many retries we've exhausted
                # We stay in strict 4 phases. Escalation happens only when phase_retries exceeds budget.
                
                if self.stagnation_level == 0:
                    self.stagnation_level = 1 # Start stagnation handling
                    self.phase_retries = 0
                elif self.phase_retries > max_retries_per_phase:
                     # Budget exhausted for current level, escalate!
                     self.stagnation_level += 1
                     self.phase_retries = 0 # Reset for new level
                     self._log(f"Escalating Stagnation Level to {self.stagnation_level} (Exhausted {max_retries_per_phase} retries)")

                # [OPTIMIZED HIERARCHY 2026-02-24: 4-Level Logic]
                strategy = "heavy_ruin" # Fallback
                
                if self.stagnation_level >= 4:
                     # Level 4: Soft Restart (The "Nuclear" Option)
                     strategy = "soft_restart"
                
                elif self.stagnation_level == 3:
                     # Level 3: Supernova Ruin (Anti-Consensus)
                     # Persistent effort to break comfortable consensus
                     strategy = "supernova_ruin"
                     
                elif self.stagnation_level == 2:
                     # Level 2: Massive Reconstructive Ruin
                     strategy = "massive_ruin"
                     
                elif self.stagnation_level == 1:
                     # Level 1: Diversification / Path Relinking
                     # Try to jump to other known elites or just shake slightly
                     if len(self.elite_pool) > 2 and random.random() < 0.6:
                         strategy = "path_relinking_to_best"
                     elif random.random() < 0.5:
                         strategy = "jump_to_secondary_peak"
                     else:
                         strategy = "heavy_ruin" 

                self._log(f"Step:{self.current_run_steps} Stagnation L{self.stagnation_level} (Try {self.phase_retries}/{max_retries_per_phase}). Qual={env.key_value:.0f} Act={strategy}")
                
                if strategy == "soft_restart":
                    self.stagnation_level = 0
                    self.phase_retries = 0
                    self.consecutive_massive_ruins = 0

                prev_val = env.key_value
                self._apply_breakout(env, strategy)
                
                # [BUG FIX 2026-02-21] Detect Hard Restart and Reset Baseline
                # If breakout resulted in a massive value drop (e.g. > 10%), it means we restarted.
                # We must reset current_best to avoid immediate stagnation detection.
                if env.key_value < current_best * 0.90:
                    self._log(f"Hard Restart Detected: Resetting Local Baseline ({current_best} -> {env.key_value})")
                    current_best = env.key_value
                    # Also reset visited stats to allow re-visiting peaks? No, keep tabu.
                    no_improve_steps = 0
                    # Reset Stagnation Level again to be safe
                    self.stagnation_level = 0
                
                # Reset counter to give the new candidate a chance
                no_improve_steps = 0
                
            # [NEW] Periodic Active Path Relinking to bridge peaks
            # Increase frequency from 300 to 100 to force more hybridization
            if self.current_run_steps % 100 == 0:
                 self._log(f"Step:{self.current_run_steps} Cur:{env.key_value} Best:{current_best} (BK:{env.best_known}) Stagnation:{no_improve_steps}")
            
            # [NEW] Periodic Active Path Relinking to bridge peaks
            # Increase frequency from 300 to 100 to force more hybridization
            # [FIX 2026-02-19] IMMUNITY PERIOD: Do NOT relink if recently restarted (within 500 steps).
            # This allows new constructive solutions to mature without being pulled back to the black hole.
            if self.current_run_steps % 100 <= 1 and len(self.elite_pool) >= 2:
                 if (self.current_run_steps - self.last_restart_step) > 500:
                     self._apply_breakout(env, "active_pool_relinking")
                 else:
                     pass # Immunized
                 
                 # Code Check Fix: Do NOT reset no_improve_steps here. 
                 # We want the main stagnation logic (Heavy Ruin/Supernova) to still trigger if this fails.
                 # no_improve_steps = 0 
                 continue

            # Sync Distributed Elite Pool periodically
            if self.current_run_steps % 50 == 0:
                self._sync_shared_pool()
                
                # Check if we are currently in a "Recovery/Exploration" phase (high no_improve_steps)
                # If we just performed a massive ruin/injection, we need time to climb back up.
                # Don't kill promising young solutions too early.
                # Only apply catch-up if we have been stagnant for a while OR if the current solution is truly abysmal for too long.
                
                # [FIX 2026-02-20] Dynamic Immunity for "Rebel" Workers (Soft Restarted)
                # If a worker recently restarted, it enters "Exploration Mode".
                # We grant it a generous grace period (e.g. 2 * node_num steps or static 2000) to find a NEW peak.
                # During this time, it is immune to "Catch-up" (being pulled back to the old peak).
                
                node_num = env.instance_data["node_num"]
                immunity_period = max(2000, node_num) # Adaptive: At least 2000, or 1x node count
                is_immune = (self.current_run_steps - self.last_restart_step) < immunity_period

                if self.elite_pool and no_improve_steps > 300:
                    pool_best_wrapper = max(self.elite_pool, key=lambda s: s["solution"].cut_value)
                    pool_best = pool_best_wrapper["solution"]
                    
                    # Original: 0.998 allowed 5321 (0.9989) to survive indefinitely.
                    # Fix: 0.9992 was too strict and caused "Ruin -> Catch-up -> Reset" loop.
                    # Relaxed to 0.90 to allow deep exploration/ruin strategies to work.
                    catch_up_threshold = 0.90 
                    
                    # Logically: If NOT immune AND score is too low -> Catch up
                    if not is_immune and env.key_value < pool_best.cut_value * catch_up_threshold:
                        from src.problems.max_cut.components import Solution
                        self._log(f"AGGRESSIVE CATCH-UP: Abandoning {env.key_value} for {pool_best.cut_value} (Threshold: {catch_up_threshold})...")
                        
                        env.current_solution = Solution(set(pool_best.set_a), set(pool_best.set_b), pool_best.cut_value)
                        # Restore History
                        env.recordings = list(pool_best_wrapper.get("history", []))

                        env.current_solution.cut_value = env.get_key_value(env.current_solution)
                        env.problem_state = env.get_problem_state()
                        
                        current_best = env.key_value
                        no_improve_steps = 0
                    elif is_immune and env.key_value < pool_best.cut_value * catch_up_threshold:
                        # Log sparsely
                        if self.current_run_steps % 500 == 0:
                             self._log(f"Catch-up IMMUNITY: Worker exploring ({self.current_run_steps - self.last_restart_step}/{immunity_period} steps). Val={env.key_value}")

