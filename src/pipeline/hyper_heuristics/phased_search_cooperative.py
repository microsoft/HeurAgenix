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
            "anti_consensus": ["anti_consensus_perturbation"]
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
            "anti_consensus": os.path.join(base_path, ruin_path, "anti_consensus_perturbation.py")
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

    def _save_to_shared_pool(self, solution, is_keep_alive=False):
        if not self.shared_pool_dir: return
        
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
            
            timestamp_int = int(current_time)
            # Filename: sol_{value}_{timestamp}_{worker}_{rand}.pkl
            filename = f"sol_{solution.cut_value}_{timestamp_int}_{self.worker_id}_{random.randint(1000,9999)}.pkl"
            filepath = os.path.join(shard_path, filename)
            
            # Atomic write
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                pickle.dump(solution, f)
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
        buckets_to_scan.append(self._get_time_bucket_path(current_time))
        # Previous hour
        buckets_to_scan.append(self._get_time_bucket_path(current_time - 3600))
        
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
                    sol = pickle.load(f)
                    self._add_to_local_pool(sol, share=False)
            except:
                pass
                
        # 2. WRITE (Keep-Alive): Periodically re-upload top elites to current bucket
        # Do this roughly every 5th sync (e.g. every ~250 steps)
        # Using simple random check 20%
        if random.random() < 0.2 and self.elite_pool:
            # Pick top 3 from local pool
            # Sort local pool by value descending
            sorted_pool = sorted(self.elite_pool, key=lambda s: s.cut_value, reverse=True)
            top_elites = sorted_pool[:3]
            
            for elite in top_elites:
                self._save_to_shared_pool(elite, is_keep_alive=True)

    def _add_to_local_pool(self, solution_obj, share=True):
        # [DYNAMIC TABU STRATEGY 2026-02-19]
        # Check against dynamic tabu list
        # If this solution belongs to an "Exhausted Basin", we reject it to prevent re-infection.
        
        # We need to access the dynamic visited_peaks from the instance.
        # Initialize if not present (defensive programming)
        if not hasattr(self, "visited_peaks"):
             self.visited_peaks = {}
             
        TABU_TOLERANCE = 5.0
        MAX_VISITS_PER_PEAK = 3
        
        for peak_val, count in self.visited_peaks.items():
            if count >= MAX_VISITS_PER_PEAK and abs(solution_obj.cut_value - peak_val) < TABU_TOLERANCE:
                # This peak is officially exhausted/tabu. Reject entry.
                # self._log(f"Rejected solution {solution_obj.cut_value} (Exhausted Basin)")
                return

        # Add copy of solution to pool
        from src.problems.max_cut.components import Solution
        
        # Deep copy the sets
        new_sol = Solution(set(solution_obj.set_a), set(solution_obj.set_b), solution_obj.cut_value)
        
        # [IMPROVED DIVERSITY CONTROL 2026-02-21]
        # Instead of just appending best values (which leads to homogenization),
        # we enforce spatial diversity based on Hamming Distance.
        
        # Helper: Calculate Hamming Distance
        def calc_dist_pool(s1, s2):
            d1 = len((s1.set_a & s2.set_b) | (s1.set_b & s2.set_a))
            d2 = len((s1.set_a & s2.set_a) | (s1.set_b & s2.set_b))
            return min(d1, d2)

        node_num = len(new_sol.set_a) + len(new_sol.set_b)

        # ------------------------------------------------------------------
        # Strategy 1: Strict Spatial Exclusion (Prevent Clones)
        # ------------------------------------------------------------------
        # We define a "Similarity Radius". Any solution within this radius
        # is considered to belong to the same "Peak".
        # Dynamic threshold: e.g. 5% of nodes. 
        # For N=2000, threshold=100. For N=800, threshold=40.
        SIMILARITY_RATIO = 0.05
        similarity_threshold = max(20, int(node_num * SIMILARITY_RATIO))

        # 1. Check if this solution belongs to an existing "Family" (Peak) in the pool
        closest_neighbor = None
        min_dist = float('inf')
        closest_index = -1
        
        for i, existing in enumerate(self.elite_pool):
            d = calc_dist_pool(new_sol, existing)
            if d < min_dist:
                min_dist = d
                closest_neighbor = existing
                closest_index = i
        
        # Policy A: If very close to an existing solution (Same Peak)
        if closest_neighbor and min_dist < similarity_threshold:
            # Only update if strictly better.
            # We want to keep the PEAK of this family.
            if new_sol.cut_value > closest_neighbor.cut_value:
                # Upgrade the existing slot to the better version
                self.elite_pool[closest_index] = new_sol
                if share: self._save_to_shared_pool(new_sol)
                # self._log(f"Pool Updated: Improved existing peak (Dist={min_dist}, Val={new_sol.cut_value})")
                return
            else:
                # We already have a better or equal representative for this peak. REJECT.
                # Even if it's equal, we reject to avoid churn without gain.
                return

        # ------------------------------------------------------------------
        # Strategy 2: Diversity Injection (Manage Pool Health)
        # ------------------------------------------------------------------ 
        # Policy B: It is a distinct solution (Distant from everyone else)
        
        # If pool is not full, just add it.
        if len(self.elite_pool) < 20:
            self.elite_pool.append(new_sol)
            if share: self._save_to_shared_pool(new_sol)
            return

        # If pool is full, we need to decide who to evict.
        # Standard logic: Check against the worst solution.
        min_val = min(s.cut_value for s in self.elite_pool)
        
        # Case 1: Better than worst (Standard quality improvement)
        if new_sol.cut_value > min_val:
            # [FIX 2026-02-26] Preventing Pool Homogenization
            # Check if we already have too many solutions with the SAME cut_value as this new one.
            # If we have >= 3 solutions with this exact score, we reject adding another one (even if it improves the worst), 
            # UNLESS the worst one is ALSO of this same score (which means we are just cycling).
            # This forces the pool to keep lower-quality but diverse solutions.
            
            same_score_count = sum(1 for s in self.elite_pool if abs(s.cut_value - new_sol.cut_value) < 1e-6)
            MAX_SAME_SCORE = 3
            
            if same_score_count >= MAX_SAME_SCORE:
                 # Too many identical scores. Reject to preserve diversity of lower scores.
                 # Exception: If the solution to be replaced (min_val) is actually much worse, 
                 # we might still want to replace it? 
                 # No, strict diversity. If we have 3 copies of Best, we don't need a 4th. 
                 # We need the 4th slot for a 2nd Best or 3rd Best to bridge the gap.
                 # self._log(f"Pool Reject: Too many solutions with score {new_sol.cut_value}")
                 return

            # Replace the worst one
            for i, s in enumerate(self.elite_pool):
                if s.cut_value == min_val:
                    self.elite_pool[i] = new_sol
                    break
            if share: self._save_to_shared_pool(new_sol)
            return
        
        # Case 2: Equal to worst
        elif new_sol.cut_value == min_val:
             # Since we passed Policy A, we know it is DISTANT from everyone (including the worst one).
             # So we have a tie in value, but new_sol offers new genes.
             # ALWAYS Replace the old worst with this new distinct one to improve diversity.
            for i, s in enumerate(self.elite_pool):
                if s.cut_value == min_val:
                    self.elite_pool[i] = new_sol
                    if share: self._save_to_shared_pool(new_sol)
                    break
            return

        # Case 3: Worse than worst (Refuse, unless it is a "Stranger")
        else:
             # Policy C: "Stranger" Admission
             # If a solution is significantly different from the ENTIRE pool, 
             # we might admit it even if it's poor, to break stagnation.
             
             # Dynamic threshold for "Stranger": e.g. 15% of nodes
             STRANGER_RATIO = 0.15
             stranger_threshold = max(30, int(node_num * STRANGER_RATIO))
             
             if min_dist > stranger_threshold:
                 # It is a stranger! 
                 # We want to add it, but we must evict someone.
                 # To maintain average quality, we should evict the worst logic.
                 # But we just established new_sol < min_val. So we are lowering the bar.
                 # We do this probabilistically to avoid flooding.
                 
                 if random.random() < 0.2:
                     # Find the worst elite to replace
                     # (We could also replace the 'most redundant' elite, but that's expensive to compute)
                     for i, s in enumerate(self.elite_pool):
                        if s.cut_value == min_val:
                            self.elite_pool[i] = new_sol
                            if share: self._save_to_shared_pool(new_sol)
                            # self._log(f"Diversity Injection: Accepted DISTANT poor solution (Val={new_sol.cut_value}, Dist={min_dist})")
                            break

    def _calculate_consensus_flip(self, node_num, flip_ratio=0.1):
        """
        Identify variables that are 'static' across the entire elite pool (Consensus),
        and select a subset of them to force flip.
        This implements the 'Anti-Consensus' or 'Non-local Move' strategy.
        """
        if not self.elite_pool:
             return []
             
        pool_size = len(self.elite_pool)
        
        # Count occurrence of each node in set_a
        # set_a_counts[i] = number of elite solutions where node i is in set_a
        set_a_counts = {}
        
        for sol in self.elite_pool:
            for node in sol.set_a:
                set_a_counts[node] = set_a_counts.get(node, 0) + 1
                
        static_nodes = []
        
        for node in range(node_num):
            count = set_a_counts.get(node, 0)
            # If node is in A for all solutions (count == pool_size) 
            # OR in A for 0 solutions (count == 0, meaning always in B)
            # It is a Static/Consensus node.
            if count == pool_size or count == 0:
                static_nodes.append(node)
                
        if not static_nodes:
            # Fallback if no consensus (unlikely in stagnation)
            return random.sample(range(node_num), int(node_num * flip_ratio))
            
        # Select a subset of static nodes to flip
        flip_count = max(1, int(len(static_nodes) * flip_ratio))
        return random.sample(static_nodes, flip_count)

    def _update_elite_pool(self, solution_obj):
        # Wrapper for backward compatibility or clarity
        self._add_to_local_pool(solution_obj, share=True)
    
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
                
                # 2. Run Heuristic (In-Place Modification)
                try:
                    env.run_heuristic(heuristic)
                except Exception as e:
                    self._log(f"Error running heuristic: {e}")
                    env.current_solution = backup_sol
                    continue

                # 3. Acceptance Criteria: Strict Ascent
                # If Score Dropped or Equal -> Revert (We want to find peaks, not drift)
                if env.key_value <= start_val:
                    # Revert
                    env.current_solution = backup_sol
                    # Restore env properties just in case
                    env.current_solution.cut_value = start_val
                    env.problem_state = env.get_problem_state() 
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
            nodes_to_flip = self._calculate_consensus_flip(node_num, flip_ratio=ratio) 
            if nodes_to_flip:
                 # Replaced direct modification with Operator
                 to_a = [n for n in nodes_to_flip if n in env.current_solution.set_b]
                 to_b = [n for n in nodes_to_flip if n in env.current_solution.set_a]
                 op = BatchInsertNodeOperator(to_a, to_b)
                 env.run_operator(op)
                 
                 self._log(f"Supernova Ruin applied: Flipped {len(nodes_to_flip)} static nodes (Ratio: {ratio:.2f}).")
            else:
                 # Fallback if no static nodes found
                 self._apply_breakout(env, "heavy_ruin")

        elif strategy == "active_pool_relinking":
            # [NEW] Active strategy: Force path relinking between distant elites
            if len(self.elite_pool) < 2:
                 return
            
            # 1. Find Best Known
            best_sol = max(self.elite_pool, key=lambda s: s.cut_value)
            
            # 2. Find a "Distant" High-Quality Elite
            # [FIX] Lower threshold to 0.97 to ensure our current elites (2400-2406) can participate
            # 2446 * 0.97 = 2372, so 2400+ are valid candidates
            candidates = [s for s in self.elite_pool if s.cut_value > env.best_known * 0.97]
            if not candidates: 
                 return
                 
            # Helper for distance
            def calc_dist(s1, s2):
                d1 = len((s1.set_a & s2.set_b) | (s1.set_b & s2.set_a))
                d2 = len((s1.set_a & s2.set_a) | (s1.set_b & s2.set_b))
                return min(d1, d2)
            
            # Find candidate farthest from best_sol
            # [FIX] Use weighted random choice to avoid "Groundhog Day" repeating the same link
            # Pick from top 3 farthest
            candidates.sort(key=lambda s: calc_dist(s, best_sol), reverse=True)
            top_candidates = candidates[:min(3, len(candidates))]
            distant_elite = random.choice(top_candidates)
            
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
                 
                 # Load best solution
                 env.current_solution = copy.deepcopy(best_sol)
                 env.current_solution.cut_value = best_sol.cut_value
                 
                 # Perturb 2% of nodes (enough to move away ~60 nodes in 3000)
                 # This is lighter than Level 1 Stagnation (Light Ruin), keeping us in the same "Peak Family".
                 micro_flip_count = max(5, int(node_num * 0.02))
                 
                 nodes_to_flip = random.sample(range(node_num), micro_flip_count)
                 to_a = [n for n in nodes_to_flip if n in env.current_solution.set_b]
                 to_b = [n for n in nodes_to_flip if n in env.current_solution.set_a]
                 op = BatchInsertNodeOperator(to_a, to_b)
                 env.run_operator(op)
                 
                 return

            self._log(f"*** ACTIVE RELINKING: Best({best_sol.cut_value}) <-> Distant({distant_elite.cut_value}, Dist={dist}) ***")

            # 3. Reset to Best, Target = Distant
            env.current_solution = copy.deepcopy(best_sol)
            env.current_solution.cut_value = best_sol.cut_value
            
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
             best_val = max(s.cut_value for s in self.elite_pool)
             best_solutions = [s for s in self.elite_pool if s.cut_value == best_val]
             
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
            env.algorithm_data["elite_pool"] = self.elite_pool
            env.run_heuristic(h, parameters={"intensity": 0.3})
            
        elif strategy == "jump_to_secondary_peak":
             # Strategy: Teleport to a high-quality local optimum
             if not self.elite_pool:
                 self._apply_breakout(env, "supernova_ruin")
                 return

             best_val = max(s.cut_value for s in self.elite_pool)
             # Candidates: High quality but strictly less than Best Known (to find secondary peaks)
             # We want to revisit peaks like 26992 to see if we can sharpen them
             # [FIX] Use relative threshold for large-weight instances (imgseg)
             threshold = max(150, env.best_known * 0.02)
             candidates = [s for s in self.elite_pool if s.cut_value >= env.best_known - threshold and s.cut_value < best_val]
             
             desc = "SECONDARY PEAK"

             # [FIX] If no secondary peak, jump to a DISTANT parallel peak (same best value)
             if not candidates:
                 def calc_dist_j(s1, s2):
                    d1 = len((s1.set_a & s2.set_b) | (s1.set_b & s2.set_a))
                    d2 = len((s1.set_a & s2.set_a) | (s1.set_b & s2.set_b))
                    return min(d1, d2)
                 
                 # Look for solutions with SAME best value but Distance > 400
                 candidates = [s for s in self.elite_pool if abs(s.cut_value - best_val) <= 1e-3 and calc_dist_j(s, env.current_solution) > 400]
                 desc = "PARALLEL UNIVERSE PEAK"
             
             if candidates:
                 target = random.choice(candidates)
                 # Deep copy
                 from src.problems.max_cut.components import Solution
                 new_sol = Solution(set(target.set_a), set(target.set_b), target.cut_value)
                 env.current_solution = new_sol
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
                noise_nodes = random.sample(range(node_num), int(node_num * ratio_noise))
                # Flip them
                to_a_noise = [n for n in noise_nodes if n in env.current_solution.set_b]
                to_b_noise = [n for n in noise_nodes if n in env.current_solution.set_a]
                op_noise = BatchInsertNodeOperator(to_a_noise, to_b_noise)
                env.run_operator(op_noise)
                self._log(f"Noise Injection: Flipped {len(noise_nodes)} nodes ({ratio_noise:.1%}) to escape basin.")
                
                # Force update value
                cur_val = env.get_key_value(env.current_solution)
                if env.current_solution.cut_value != cur_val:
                    env.current_solution.cut_value = cur_val

            else:
                 # Fallback: Random Flip 40%
                 nodes = random.sample(range(node_num), int(node_num * 0.40))
                 
                 to_a = [n for n in nodes if n in env.current_solution.set_b]
                 to_b = [n for n in nodes if n in env.current_solution.set_a]
                 op = BatchInsertNodeOperator(to_a, to_b)
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
                  sorted_elites = sorted(self.elite_pool, key=lambda s: calc_dist_r(s, env.current_solution), reverse=True)
                  # Pick from top 5 furthest
                  target = random.choice(sorted_elites[:5])
                  
                  # [CRITICAL FIX 2026-02-19] Check if the "furthest" elite is actually distant.
                  # If the pool has collapsed (Homogenized), the furthest elite might be just 10 flips away.
                  # In that case, a Soft Restart is useless. We must force a HARD RESTART (Constructive).
                  dist = calc_dist_r(target, env.current_solution)
                  min_restart_dist = max(50, int(node_num * 0.05)) # e.g. 150 nodes for 3000 node graph
                  
                  if dist < min_restart_dist:
                       self._log(f"Soft Restart Aborted: Pool Homogenized (Max Dist={dist} < {min_restart_dist}). Forcing Hard Constructive Restart.")
                       force_constructive = True
                  else:
                       from src.problems.max_cut.components import Solution
                       env.current_solution = Solution(set(target.set_a), set(target.set_b), target.cut_value)
                       self._log(f"Restarted from Distant Elite (Val: {target.cut_value}, Dist: {dist})")
             
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
                  
                  if not env.is_complete_solution:
                      # Total random fallback if heuristics fail
                      nodes = list(range(node_num))
                      random.shuffle(nodes)
                      mid = node_num // 2
                      env.current_solution = Solution(set(nodes[:mid]), set(nodes[mid:]), 0)
                      env.current_solution.cut_value = env.get_key_value(env.current_solution)

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
        self._update_elite_pool(env.current_solution)
        
        no_improve_steps = 0

        # CHANGE: Use instance variable to track steps for coordination with restart logic
        self.current_run_steps = 0
        
        self._log(f"Starting Breakout Search from {current_best}...")

        # [DYNAMIC TABU LIST]
        # Keep track of local optima we have visited frequently.
        # Format: {cut_value: visit_count}
        self.visited_peaks = {}
        TABU_TOLERANCE = 5.0 # Solutions within this range are considered the "same" peak
        MAX_VISITS_PER_PEAK = 3 # How many times can we rediscover the same peak before banning it?

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
                self._apply_breakout(env, "supernova_ruin")
                # Don't reset steps, we want to keep pressure high until we leave
                continue

            # --- Phase A: Repair / Improve ---
            # Try to improve current solution (which might be ruined)
            improved = self._run_improvement_phase(env)
            
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
                self._update_elite_pool(env.current_solution)
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
                    
                    if (current_best - saved_best) > 1e-6:
                        env.dump_result(result_file=f"breakthrough_from_worker_{self.worker_id}_{current_best}.txt")

                elif abs(current_best - env.best_known) < 1e-6:
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
                    self._update_elite_pool(env.current_solution)

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
                    pool_best = max(self.elite_pool, key=lambda s: s.cut_value)
                    
                    # Original: 0.998 allowed 5321 (0.9989) to survive indefinitely.
                    # Fix: 0.9992 was too strict and caused "Ruin -> Catch-up -> Reset" loop.
                    # Relaxed to 0.90 to allow deep exploration/ruin strategies to work.
                    catch_up_threshold = 0.90 
                    
                    # Logically: If NOT immune AND score is too low -> Catch up
                    if not is_immune and env.key_value < pool_best.cut_value * catch_up_threshold:
                        from src.problems.max_cut.components import Solution
                        self._log(f"AGGRESSIVE CATCH-UP: Abandoning {env.key_value} for {pool_best.cut_value} (Threshold: {catch_up_threshold})...")
                        
                        env.current_solution = Solution(set(pool_best.set_a), set(pool_best.set_b), pool_best.cut_value)
                        env.current_solution.cut_value = env.get_key_value(env.current_solution)
                        env.problem_state = env.get_problem_state()
                        
                        current_best = env.key_value
                        no_improve_steps = 0
                    elif is_immune and env.key_value < pool_best.cut_value * catch_up_threshold:
                        # Log sparsely
                        if self.current_run_steps % 500 == 0:
                             self._log(f"Catch-up IMMUNITY: Worker exploring ({self.current_run_steps - self.last_restart_step}/{immunity_period} steps). Val={env.key_value}")

