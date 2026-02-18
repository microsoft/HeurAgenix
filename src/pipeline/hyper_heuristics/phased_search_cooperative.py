import os
import random
import pickle
import glob
import uuid
import time
import hashlib
import copy
from datetime import datetime, timedelta
from src.problems.base.env import BaseEnv
from src.util.util import load_function
from src.pipeline.hyper_heuristics.phased_search_adaptive_polishing import PhasedSearchAdaptivePolishingHyperHeuristic
from src.problems.max_cut.components import BatchInsertNodeOperator, BatchDeleteOperator, Solution

class PhasedSearchCooperativeHyperHeuristic(PhasedSearchAdaptivePolishingHyperHeuristic):
    def __init__(self, heuristic_pool, problem, shared_pool_dir=None, top_k=10, load_ratio=1.0, fail_fast_threshold=0.02, initial_solution_paths=None, worker_id=None):
        # Force load_ratio to 1.0 to ensure we always try to load
        self.shared_pool_dir = shared_pool_dir
        super().__init__(heuristic_pool, problem, shared_pool_dir, top_k, 1.0, fail_fast_threshold)
        
        self.elite_pool = []
        self.breakout_heuristics = {}
        self._load_breakout_heuristics()
        self.stagnation_level = 0 # Track escalation level
        
        # Distributed Cooperation Setup
        if worker_id is not None:
            self.worker_id = str(worker_id)
        else:
            self.worker_id = str(uuid.uuid4())[:8]
            
        # Hash worker_id to get a shard index (0-9)
        self.shard_id = int(hashlib.md5(self.worker_id.encode()).hexdigest(), 16) % 10
        # self.shared_pool_dir = None # REMOVED BUGGY LINE
        
        # Throttling
        self.last_upload_time = 0
        self.last_upload_value = 0
        self.last_sync_time = 0
        
        if self.shared_pool_dir:
            # Use the path provided explicitly by search_best.py.
            # No more complex inference or high_quality_solution logic.
            try:
                self._log(f"Shared Elite Pool Directory: {self.shared_pool_dir}")
                os.makedirs(self.shared_pool_dir, exist_ok=True)
            except OSError:
                pass 
        
        # [NEW] Initial Load Logic from Main Process
        # Use paths provided by the main process (which filtered them by diversity)
        if initial_solution_paths:
             self._load_initial_solutions_from_paths(initial_solution_paths)
        else:
            # Fallback legacy load
            # self._load_initial_pool_from_disk()
            pass

    def _load_initial_solutions_from_paths(self, paths):
        self._log(f"Loading {len(paths)} initial diverse solutions from main process...")
        count = 0
        for path in paths:
            try:
                if not os.path.exists(path):
                    continue
                with open(path, 'rb') as f:
                    sol = pickle.load(f)
                    # We use _add_to_local_pool but share=False because these are already known elites/breakthroughs
                    # We do NOT share them back immediately to avoid storm
                    self._add_to_local_pool(sol, share=False)
                    count += 1
            except Exception as e:
                self._log(f"Warning: Failed to load {path}: {e}")
        self._log(f"Successfully loaded {count} elites locally.")

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
        print(f"[{datetime.now().strftime('%H:%M:%S')}, Worker:{self.worker_id}] {message}", flush=True)

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
        # Add copy of solution to pool
        from src.problems.max_cut.components import Solution
        
        # Deep copy the sets
        new_sol = Solution(set(solution_obj.set_a), set(solution_obj.set_b), solution_obj.cut_value)
        
        # Check uniqueness by value AND diversity (Hamming Distance)
        # If we already have this EXACT solution (Hamming Dist = 0), skip it to prevent cloning
        is_duplicate = False
        
        # Safer node_num inferrence
        # If we can't trust self.problem, we assume partition is complete
        node_num = len(new_sol.set_a) + len(new_sol.set_b)
        
        for existing in self.elite_pool:
            if existing.cut_value == new_sol.cut_value:
                # Check for identical sets (Hamming dist = 0)
                # MaxCut solution is defined by set_a/set_b partition.
                # Need to check if new_sol.set_a == existing.set_a OR new_sol.set_a == existing.set_b (symmetric)
                # Assuming sets are normalized or checking both
                if new_sol.set_a == existing.set_a or new_sol.set_a == existing.set_b:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            if len(self.elite_pool) < 20:
                self.elite_pool.append(new_sol)
            else:
                # Replace random or worst?
                # If new solution is better than worst, replace worst
                # stored sorted? No.
                min_val = min(s.cut_value for s in self.elite_pool)
                if new_sol.cut_value > min_val:
                    # Find index of min
                    for i, s in enumerate(self.elite_pool):
                        if s.cut_value == min_val:
                            self.elite_pool[i] = new_sol
                            break
                else:
                    # If same quality but different structure? Keep diversity
                    # If value equals min_val, but it's different structure, replace one with prob
                    # [FIX] If it's a Top Tier solution (equal to max/BK), accept it with high probability (1.0) to encourage "Parallel Peak" drift
                    # Otherwise use low probability
                    is_top_tier = (new_sol.cut_value == min_val and min_val == max(s.cut_value for s in self.elite_pool))
                    accept_prob = 1.0 if is_top_tier else 0.3
                    
                    if new_sol.cut_value == min_val and random.random() < accept_prob:
                         for i, s in enumerate(self.elite_pool):
                            if s.cut_value == min_val:
                                self.elite_pool[i] = new_sol
                                break
        
            if share:
                self._save_to_shared_pool(new_sol)

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

    def _load_breakout_heuristics(self):
        # Explicitly load the designated heuristics for breakout
        # Using relative paths from the workspace root or absolute paths
        # load_function expects path relative to src/problems/max_cut/heuristics/ usually, or just filename if in path
        
        # We'll use the full path loader logic from util.py if possible, or manually load
        # For safety/simplicity, I will try to load by filename if they are in the standard folders
        
        base_path = "src/problems/max_cut/heuristics"
        ruin_path = "evolved_heuristics.part3"
        
        h_map = {
            "batch_cluster_ruin": os.path.join(base_path, ruin_path, "batch_cluster_ruin.py"),
            "batch_worst_ruin": os.path.join(base_path, ruin_path, "batch_worst_ruin.py"),
            "path_relinking": os.path.join(base_path, "path_relinking_guided_perturbation.py"),
            "anti_consensus": os.path.join(base_path, ruin_path, "anti_consensus_perturbation.py")
        }
        
        for key, path in h_map.items():
            try:
                # Assuming running from root
                if os.path.exists(path):
                    self.breakout_heuristics[key] = load_function(path, problem=self.problem)
                else:
                    # Try absolute path
                    abs_path = os.path.join(os.getcwd(), path)
                    if os.path.exists(abs_path):
                        self.breakout_heuristics[key] = load_function(abs_path, problem=self.problem)
                    else:
                        self._log(f"Warning: Could not find breakout heuristic {key} at {path}")
            except Exception as e:
                self._log(f"Error loading {key}: {e}")


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
            threshold = max(10, int(node_num * 0.015))

            if dist < threshold: 
                 self.stagnation_level += 2
                 self._log(f"Active Relinking: Target too close (Dist={dist} < Threshold={threshold}). Accelerating Stagnation Level +2.")
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
                from src.problems.max_cut.components import BatchInsertNodeOperator
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
                 
                 from src.problems.max_cut.components import BatchInsertNodeOperator
                 to_a = [n for n in nodes if n in env.current_solution.set_b]
                 to_b = [n for n in nodes if n in env.current_solution.set_a]
                 op = BatchInsertNodeOperator(to_a, to_b)
                 env.run_operator(op)
                 
                 self._log(f"Fallback Ruin: Random Flipped {len(nodes)} nodes.")


    def run(self, env: BaseEnv) -> bool:
        # [REFACTORED for Cooperative Search]
        # 1. Try to load initial solution (Hot Start) from the Shared Pool
        # We rely SOLELY on the shared pool (which might have been pre-populated or filled by other workers)
        loaded = False
        
        # Explicitly maximize chances by syncing first
        self._sync_shared_pool()

        # Try to find the best available solution in our local view of the pool
        if self.elite_pool:
            # Stochastic Hot Start: Pick randomly from Top K elite solutions
            # This prevents all workers from greedily converging on the same local max (Black Hole Effect)
            
            # Filter for unique scores to ensure diversity
            unique_pool = []
            seen_scores = set()
            for sol in sorted(self.elite_pool, key=lambda s: s.cut_value, reverse=True):
                if sol.cut_value not in seen_scores:
                    unique_pool.append(sol)
                    seen_scores.add(sol.cut_value)
            
            # If we have enough unique solutions, pick from top 10. Otherwise, broaden search.
            search_space = unique_pool[:min(len(unique_pool), 10)]
            if len(search_space) < 3 and len(self.elite_pool) > 20:
                 # Fallback: If pool is dominated by duplicates, broaden to raw top 50 to find *any* deviation
                 search_space = sorted(self.elite_pool, key=lambda s: s.cut_value, reverse=True)[:50]
            
            best_sol = random.choice(search_space)
            
            self._log(f"Hot Start: Loaded stochastic best from Elite Pool (Val: {best_sol.cut_value} | Pool Max: {max(p.cut_value for p in self.elite_pool)})")
            
            # Deep Copy to ensure safety
            from src.problems.max_cut.components import Solution
            env.current_solution = Solution(set(best_sol.set_a), set(best_sol.set_b), best_sol.cut_value)
            
            # Update Environment State
            env.current_solution.cut_value = env.get_key_value(env.current_solution) # Recalculate to be safe
            env.problem_state = env.get_problem_state()
            
            # Update Env Best Known if we accidentally loaded something better
            if env.key_value > env.best_known:
                env.best_known = env.key_value
                
            loaded = True
        
        if not loaded: 
            self._log("No Elite Pool solutions found. Switching to Constructive Phase (Cold Start)...")
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
        total_steps = 0
        
        self._log(f"Starting Breakout Search from {current_best}...")

        while env.continue_run:
            total_steps += 1
            
            # --- Phase A: Repair / Improve ---
            # Try to improve current solution (which might be ruined)
            improved = self._run_improvement_phase(env)
            
            # --- Phase B: Check Status ---
            # 1. Update Global/Local Best
            if env.key_value > current_best:
                current_best = env.key_value
                no_improve_steps = 0
                self.stagnation_level = 0
                self._update_elite_pool(env.current_solution)
                self._log(f"Step:{total_steps} NEW LOCAL BEST: {current_best}")
                
                # Always dump intermediate improvements as TXT for easy reuse
                env.dump_result(result_file=f"intermediate_result.{current_best}.txt")

                if current_best > env.best_known:
                     self._log(f"!!! BREAKTHROUGH: {current_best} > {env.best_known} !!!")
                     env.best_known = current_best
                     env.dump_result(result_file=f"breakthrough.{current_best}.txt")
                     return True
            else:
                no_improve_steps += 1
                
                # 2. Relaxed Elite Pool Update (Fix for Path Relinking)
                # If we are stuck but the solution is still decent (e.g. > 98% of BK)
                # we add it to the pool to provide diversity for path relinking.
                # Don't add every step, maybe every 10 steps to avoid flooding with identical copies
                is_best_known = env.key_value >= env.best_known
                if is_best_known or (env.key_value >= env.best_known * 0.98 and total_steps % 10 == 0):
                    self._update_elite_pool(env.current_solution)

            # --- Phase C: Breakout / Ruin Strategies ---
            # Adaptive Patience: 
            # If we are close to best known, be more patient with small moves.
            # If we are far (after ruin), be impatient.
            
            # Adaptive Patience based on problem size
            # For 1000 nodes, 50-100 is okay. For 3000 nodes, we need 200-300.
            patience = max(50, int(env.instance_data["node_num"] / 10))
            
            if no_improve_steps > patience:
                # Escalation using stagnation_level
                self.stagnation_level += 1
                
                strategy = "light_ruin"
                
                # Check if we have a "Good Local Optima" that is worth relinking before destroying
                # Condition: High quality (>99.5% BK) AND Diversity (>200 distance) exists in pool
                is_high_quality_stagnation = env.key_value > env.best_known * 0.995
                
                if self.stagnation_level >= 5:
                     # Ultimate Weapon: Massive Ruin (70%)
                     # Replaces Supernova because 10-20% was not enough for sg3dl149000
                     strategy = "massive_ruin"
                     # Reset stagnation completely to allow reconstruction from the ashes
                     self.stagnation_level = 0 
                elif self.stagnation_level >= 4:
                     # Diversity Injection: Try jumping to a different peak or heavy ruin
                     if random.random() < 0.4:
                         strategy = "jump_to_secondary_peak"
                     else:
                         strategy = "path_relinking_to_best"
                elif self.stagnation_level >= 3:
                     # If we are high quality, TRY HARD to link current state to Best Known
                     if is_high_quality_stagnation and "path_relinking" in self.breakout_heuristics:
                         strategy = "path_relinking_to_best"
                     else:
                         strategy = "heavy_ruin"
                elif self.stagnation_level >= 2:
                     strategy = "medium_ruin"
                
                self._log(f"Step:{total_steps} Stagnation (Level {self.stagnation_level}). Qual={env.key_value:.0f} Triggering {strategy}...")
                self._apply_breakout(env, strategy)
                
                # Reset counter to give the new candidate a chance
                no_improve_steps = 0
                
            # Log periodically
            if total_steps % 100 == 0:
                 self._log(f"Step:{total_steps} Cur:{env.key_value} Best:{current_best} (BK:{env.best_known}) Stagnation:{no_improve_steps}")
            
            # [NEW] Periodic Active Path Relinking to bridge peaks
            # Increase frequency from 300 to 100 to force more hybridization
            if total_steps % 100 <= 1 and len(self.elite_pool) >= 2:
                 self._apply_breakout(env, "active_pool_relinking")
                 # Code Check Fix: Do NOT reset no_improve_steps here. 
                 # We want the main stagnation logic (Heavy Ruin/Supernova) to still trigger if this fails.
                 # no_improve_steps = 0 
                 continue

            # Sync Distributed Elite Pool periodically
            if total_steps % 50 == 0:
                self._sync_shared_pool()
                
                # Check if we are currently in a "Recovery/Exploration" phase (high no_improve_steps)
                # If we just performed a massive ruin/injection, we need time to climb back up.
                # Don't kill promising young solutions too early.
                # Only apply catch-up if we have been stagnant for a while OR if the current solution is truly abysmal for too long.
                
                if self.elite_pool and no_improve_steps > 300:
                    pool_best = max(self.elite_pool, key=lambda s: s.cut_value)
                    
                    # Original: 0.998 allowed 5321 (0.9989) to survive indefinitely.
                    # Fix: 0.9992 was too strict and caused "Ruin -> Catch-up -> Reset" loop.
                    # Relaxed to 0.90 to allow deep exploration/ruin strategies to work.
                    catch_up_threshold = 0.90 
                    
                    if env.key_value < pool_best.cut_value * catch_up_threshold:
                        from src.problems.max_cut.components import Solution
                        self._log(f"AGGRESSIVE CATCH-UP: Abandoning {env.key_value} for {pool_best.cut_value} (Threshold: {catch_up_threshold})...")
                        
                        env.current_solution = Solution(set(pool_best.set_a), set(pool_best.set_b), pool_best.cut_value)
                        env.current_solution.cut_value = env.get_key_value(env.current_solution)
                        env.problem_state = env.get_problem_state()
                        
                        current_best = env.key_value
                        no_improve_steps = 0

        return False
