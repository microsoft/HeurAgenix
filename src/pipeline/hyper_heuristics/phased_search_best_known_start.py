import os
import random
import pickle
import glob
import uuid
import time
import hashlib
from datetime import datetime, timedelta
from src.problems.base.env import BaseEnv
from src.util.util import load_function
from src.pipeline.hyper_heuristics.phased_search_adaptive_polishing import PhasedSearchAdaptivePolishingHyperHeuristic

class PhasedSearchBestKnownStartHyperHeuristic(PhasedSearchAdaptivePolishingHyperHeuristic):
    def __init__(self, heuristic_pool, problem, high_quality_solution_dir=None, top_k=10, load_ratio=1.0, fail_fast_threshold=0.02):
        # Force load_ratio to 1.0 to ensure we always try to load
        self.high_quality_solution_dir = high_quality_solution_dir
        super().__init__(heuristic_pool, problem, high_quality_solution_dir, top_k, 1.0, fail_fast_threshold)
        
        self.elite_pool = []
        self.breakout_heuristics = {}
        self._load_breakout_heuristics()
        self.stagnation_level = 0 # Track escalation level
        
        # Distributed Cooperation Setup
        self.worker_id = str(uuid.uuid4())[:8]
        # Hash worker_id to get a shard index (0-9)
        self.shard_id = int(hashlib.md5(self.worker_id.encode()).hexdigest(), 16) % 10
        self.shared_pool_dir = None
        
        # Throttling
        self.last_upload_time = 0
        self.last_upload_value = 0
        self.last_sync_time = 0
        
        if self.high_quality_solution_dir:
            # User request: Store elite pool in output/max_cut/elite_pool/{instance}
            # Extract instance name from path (assumed .../g63.mc/high_quality_solution)
            
            # Determine base output dir consistent with instructions
            base_output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..", "orllm", "output") if os.getenv("AMLT_OUTPUT_DIR") else "output"

            try:
                path_parts = self.high_quality_solution_dir.split(os.sep)
                # Find the part that looks like an instance name (e.g. g63.mc)
                # It is usually the parent of 'high_quality_solution'
                if 'high_quality_solution' in path_parts:
                    idx = path_parts.index('high_quality_solution')
                    instance_name = path_parts[idx-1] # e.g. g63.mc

                    # Construct new path: output/max_cut/elite_pool/{instance_name}
                    self.shared_pool_dir = os.path.join(base_output_dir, "max_cut", "elite_pool", instance_name)
                else:
                    # Fallback: If path starts with "output", replace it with base_output_dir
                    if self.high_quality_solution_dir.startswith("output"):
                         rel_path = os.path.relpath(self.high_quality_solution_dir, "output")
                         self.shared_pool_dir = os.path.join(base_output_dir, rel_path, 'elite_pool')
                    else:
                         self.shared_pool_dir = os.path.join(self.high_quality_solution_dir, 'elite_pool')
            except:
                 # Exception Fallback
                 if self.high_quality_solution_dir.startswith("output"):
                     rel_path = os.path.relpath(self.high_quality_solution_dir, "output")
                     self.shared_pool_dir = os.path.join(base_output_dir, rel_path, 'elite_pool')
                 else:
                     self.shared_pool_dir = os.path.join(self.high_quality_solution_dir, 'elite_pool')

            # Base dir created once
            try:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Shared Elite Pool Directory: {self.shared_pool_dir}", flush=True)
                os.makedirs(self.shared_pool_dir, exist_ok=True)
            except OSError:
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
                    if new_sol.cut_value == min_val and random.random() < 0.3:
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
            "path_relinking": os.path.join(base_path, "path_relinking_guided_perturbation.py")
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
                        print(f"Warning: Could not find breakout heuristic {key} at {path}", flush=True)
            except Exception as e:
                print(f"Error loading {key}: {e}", flush=True)


    def _run_improvement_phase(self, env):
        # Run UCB or Random selection from improvement_heuristics
        if not self.improvement_heuristics: return False
        
        heuristic = random.choice(self.improvement_heuristics)
        start_val = env.key_value
        env.run_heuristic(heuristic)
        
        return env.key_value > start_val

    def _apply_breakout(self, env, strategy):
        node_num = env.instance_data["node_num"]
        
        if strategy == "supernova_ruin":
            # "Anti-Consensus" Strategy: Flip stable variables
            # Increase intensity to 10%-20% to escape deep basin
            ratio = random.uniform(0.10, 0.20)
            nodes_to_flip = self._calculate_consensus_flip(node_num, flip_ratio=ratio) 
            if nodes_to_flip:
                 env.current_solution.set_a.symmetric_difference_update(nodes_to_flip)
                 # Recalculate set_b and value
                 all_nodes = set(range(node_num))
                 env.current_solution.set_b = all_nodes - env.current_solution.set_a
                 env.current_solution.cut_value = env.get_key_value(env.current_solution)
                 env.problem_state = env.get_problem_state()
                 print(f"[{datetime.now().strftime('%H:%M:%S')}] Supernova Ruin applied: Flipped {len(nodes_to_flip)} static nodes (Ratio: {ratio:.2f}).", flush=True)
            else:
                 # Fallback if no static nodes found
                 self._apply_breakout(env, "heavy_ruin")

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
             print(f"[{datetime.now().strftime('%H:%M:%S')}] Targeted Path Relinking -> Best Known ({best_val})", flush=True)

        elif strategy == "path_relinking" and "path_relinking" in self.breakout_heuristics:
            # Parameters: intensity
            h = self.breakout_heuristics["path_relinking"]
            # Pass elite_pool via algorithm_data
            # We need to hack/inject elite_pool into algorithm_data if not present
            # env.run_heuristic passes env.problem_state and env.algorithm_data
            env.algorithm_data["elite_pool"] = self.elite_pool
            env.run_heuristic(h, parameters={"intensity": 0.3})
            
        elif strategy == "jump_to_secondary_peak":
             # Strategy: Teleport to a high-quality local optimum that is NOT the current best known
             if not self.elite_pool:
                 self._apply_breakout(env, "supernova_ruin")
                 return

             best_val = max(s.cut_value for s in self.elite_pool)
             # Candidates: High quality but strictly less than Best Known (to find secondary peaks)
             # We want to revisit peaks like 26992 to see if we can sharpen them
             candidates = [s for s in self.elite_pool if s.cut_value >= env.best_known - 150 and s.cut_value < best_val]
             
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
                 print(f"[{datetime.now().strftime('%H:%M:%S')}] *** JUMPED TO SECONDARY PEAK: {env.current_solution.cut_value} (from pool of {len(candidates)}) ***", flush=True)
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

    def run(self, env: BaseEnv) -> bool:
        # 1. Load Initial (Best Known)
        loaded, _ = self._try_load_initial_solution(env)
        if not loaded: 
            print("Failed to load best known. Aborting Strategy 2.", flush=True)
            return False
            
        current_best = env.key_value
        self._update_elite_pool(env.current_solution)
        
        no_improve_steps = 0
        total_steps = 0
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Breakout Search from {current_best}...", flush=True)

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
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Step:{total_steps} NEW LOCAL BEST: {current_best}", flush=True)
                
                if current_best > env.best_known:
                     print(f"[{datetime.now().strftime('%H:%M:%S')}] !!! BREAKTHROUGH: {current_best} > {env.best_known} !!!", flush=True)
                     env.best_known = current_best
                     env.dump_result(result_file=f"breakthrough.{current_best}.txt")
                     return True
            else:
                no_improve_steps += 1
                
                # 2. Relaxed Elite Pool Update (Fix for Path Relinking)
                # If we are stuck but the solution is still decent (e.g. > 99% of BK)
                # we add it to the pool to provide diversity for path relinking.
                # Don't add every step, maybe every 10 steps to avoid flooding with identical copies
                if env.key_value >= env.best_known * 0.99 and total_steps % 10 == 0:
                    self._update_elite_pool(env.current_solution)

            # --- Phase C: Breakout / Ruin Strategies ---
            # Adaptive Patience: 
            # If we are close to best known, be more patient with small moves.
            # If we are far (after ruin), be impatient.
            
            patience = 50 
            
            if no_improve_steps > patience:
                # Escalation using stagnation_level
                self.stagnation_level += 1
                
                strategy = "light_ruin"
                
                # Check if we have a "Good Local Optima" that is worth relinking before destroying
                # Condition: High quality (>99.5% BK) AND Diversity (>200 distance) exists in pool
                is_high_quality_stagnation = env.key_value > env.best_known * 0.995
                
                if self.stagnation_level >= 5:
                     # Ultimate Weapon: Supernova / Anti-Consensus
                     strategy = "supernova_ruin"
                     # Reset stagnation to give it time to recover
                     self.stagnation_level = 3 
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
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Step:{total_steps} Stagnation (Level {self.stagnation_level}). Qual={env.key_value:.0f} Triggering {strategy}...", flush=True)
                self._apply_breakout(env, strategy)
                
                # Reset counter to give the new candidate a chance
                no_improve_steps = 0
                
            # Log periodically
            if total_steps % 100 == 0:
                 print(f"[{datetime.now().strftime('%H:%M:%S')}] Step:{total_steps} Cur:{env.key_value} Best:{current_best} (BK:{env.best_known})", flush=True)
            
            # Sync Distributed Elite Pool periodically
            if total_steps % 50 == 0:
                self._sync_shared_pool()

        return False
        
    def _try_load_initial_solution(self, env: BaseEnv) -> tuple[bool, bool]:
        """
        Overrides the loading logic to specifically look for the best known solution file.
        Returns: (loaded_success, is_fragile_elite)
        """
        data_name = env.data_ref_name # e.g., "g81.mc"
        if data_name.endswith(".mc") or data_name.endswith(".vrp"):
            base_name = data_name.split(".")[0]
        else:
            base_name = data_name
        
        target_path = os.path.join(self.high_quality_solution_dir, f"best_known.txt")
            
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading Best Known from {target_path}...", flush=True)
        
        try:
            # Load using env.load_solution first
            if env.load_solution(target_path):
                # Check if set_b is missing and fix it
                node_num = env.instance_data["node_num"]
                all_nodes = set(range(node_num))
                
                if len(env.current_solution.set_b) == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Inferring set_b from set_a...", flush=True)
                    env.current_solution.set_b = all_nodes - env.current_solution.set_a
                    
                # Recalculate cut value just in case
                env.current_solution.cut_value = env.get_key_value(env.current_solution)
                env.problem_state = env.get_problem_state()
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Successfully loaded Best Known solution! Value: {env.key_value}", flush=True)
                
                # Force update best_known in env if our loaded solution is better (or equal)
                if env.key_value > env.best_known:
                    env.best_known = env.key_value
                    
                return True, True # True, True -> Loaded, Fragile Elite Mode
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] env.load_solution returned False.", flush=True)
                return False, False

        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error loading best known: {e}", flush=True)
            return False, False
