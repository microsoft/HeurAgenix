import os
import time
import random
import glob
import pickle
import hashlib
from src.problems.cvrp.env import Env  # Adjust import based on actual CVRP env location
from src.util.util import load_function

class PhasedSearchCvrpHyperHeuristic:
    """
    CVRP-specialized Phased Search hyper-heuristic scheduling strategy (Skeleton version).
    Maintains 100% log and architecture style compatibility with the Max-Cut Twin-Engine.
    """
    def __init__(self, heuristic_pool, problem, shared_pool_dir=None, worker_id=None, logger=None, max_restarts=None, **kwargs):
        self.heuristic_pool_names = heuristic_pool
        self.logger = logger
        self.problem = problem
        self.worker_id = str(worker_id)
        self.shared_pool_dir = shared_pool_dir
        self.max_restarts = max_restarts
        self.restart_count = 0
        
        # Heuristics lists
        self.constructive_heuristics = []
        self.improvement_heuristics = []
        self.ruin_heuristics = []
        self.breakout_heuristics = {}
        
        self._classify_heuristics()
        
        # State tracking
        self.elite_pool = []
        self.stagnation_level = 0
        self.consecutive_massive_ruins = 0
        
        # Distributed Cooperation Setup
        self.shard_id = int(hashlib.md5(self.worker_id.encode()).hexdigest(), 16) % 10
        self.pool_id = 0
        self.pool_type = 'inherit'
        self.pending_rebuild = False 
        
        self.POOL_CAPACITY = 10000
        
        # Initialize Base Pool Directory
        if self.shared_pool_dir:
            try:
                self.logger(f"Shared Elite Pool Directory: {self.shared_pool_dir}")
                os.makedirs(self.shared_pool_dir, exist_ok=True)
                # TODO: Check for existing epoch pools and sync state
            except OSError:
                pass 

    # =====================================================================
    # 1. Heuristics Classification
    # =====================================================================
    def _classify_heuristics(self):
        """
        Classify based on CVRP operator names into Constructive, Improvement, and Breakout(Ruin/Reconstruct).
        """
        # Hardcode CVRP heuristic categories
        constructive_names = {
            "nearest_neighbor_54a9",
            "nearest_neighbor_99ba",
            "saving_algorithm_710e",
            "petal_algorithm_b384",
            "greedy_f4c4",
            "farthest_insertion_4e1d",
            "farthest_insertion_6308",
            "min_cost_insertion_048f",
            "min_cost_insertion_3b2b",
            "random_bfdc",
            "regret_insertion_2f3a" # Include strong operator for tight capacities
        }
        
        improvement_names = {
            "two_opt_0554",
            "three_opt_e8d7",
            "node_shift_between_routes_7b8a",
            "variable_neighborhood_search_614b"
        }
        
        breakout_map = {
            "mass_ruin": ["radial_ruin_3c4d", "random_ruin_1a2b"],
            "recreate": ["regret_insertion_2f3a", "min_cost_insertion_048f"],
            "crossover": ["route_based_crossover_9f8a"]
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
        
    def _get_pool_path(self, pool_id, pool_type):
        """Returns the directory path for a specific Epoch Pool."""
        if not self.shared_pool_dir: return ""
        return os.path.join(self.shared_pool_dir, f"pool_{pool_id}_{pool_type}")
        
    def _get_shard_path(self, bucket_path, shard_index):
        return os.path.join(bucket_path, f"shard_{shard_index}")

    # =====================================================================
    # 3. Core Solving Phases Strategy
    # =====================================================================
    def _run_improvement_phase(self, env):
        """
        Phase B: Vigorous Hill Climbing (VND)
        Squeeze the solution using Improvement operators continuously until hitting the valley floor.
        """
        if not self.improvement_heuristics: return False
        
        # Max VND loops to guarantee convergence prevention of infinite drift
        max_vnd_loops = 100000 
        total_improved = False
        
        heuristics_queue = list(self.improvement_heuristics)
        
        for loop_idx in range(max_vnd_loops):
            improved_in_this_loop = False
            random.shuffle(heuristics_queue)
            
            for heuristic in heuristics_queue:
                # 1. Snapshot State
                backup_wrapper = env.export_solution_wrapper()
                # Use standard wrapper logic (for CVRP, cut_value actually holds cost when converted,
                # but base env export handles this by dumping to dict)
                # Wait, Env.export_solution_wrapper returns a dict with 'solution'. For CVRP, it's Total Cost.
                start_val = backup_wrapper["solution"].total_cost
                
                # 2. Run Heuristic
                try:
                    env.run_heuristic(heuristic)
                except Exception as e:
                    env.import_solution_wrapper(backup_wrapper)
                    continue

                # 3. Acceptance Criteria: Strict Descent (CVRP is Min Problem)
                # If Cost Increased or Stayed Equal -> Revert
                if env.key_value >= start_val - 1e-4:
                    env.import_solution_wrapper(backup_wrapper)
                else:
                    # Accepted
                    improved_in_this_loop = True
                    total_improved = True
            
            # If a full pass yielded no gain, we are at a local optimum.
            if not improved_in_this_loop:
                break
                
        return total_improved
        

    def _scan_pool_epochs(self):
        """Scans the shared directory for pool_{id}_{type} folders and returns the latest epoch."""
        if not self.shared_pool_dir: return (self.pool_id, self.pool_type)
        try:
            entries = os.listdir(self.shared_pool_dir)
            pools = []
            for entry in entries:
                parts = entry.split('_')
                if len(parts) >= 3 and parts[0] == 'pool' and parts[1].isdigit():
                    pid = int(parts[1])
                    ptype = parts[2]
                    path = os.path.join(self.shared_pool_dir, entry)
                    if os.path.isdir(path):
                        pools.append((pid, ptype))
            if pools:
                pools.sort(key=lambda x: x[0])
                return pools[-1]
            return (self.pool_id, self.pool_type)
        except Exception as e:
            self.logger(f"Error scanning pool epochs: {e}")
            return (self.pool_id, self.pool_type)

    def _check_and_update_pool_id(self):
        """Scans periodically and updates the pool pointer if a new epoch is found."""
        old_id = self.pool_id
        new_id, new_type = self._scan_pool_epochs()
        if new_id > old_id:
            self.pool_id = new_id
            self.pool_type = new_type
            self.logger(f"-> [FOLLOW] Detected REVOLUTION (Pool {new_id}). Resetting...")
            self.pending_rebuild = True

    def _get_cvrp_fingerprint(self, env: Env):
        """
        Creates a set of all edges (u,v) in the CVRP solution.
        u < v to make them un-directed edges.
        """
        edges = set()
        if hasattr(env, 'current_solution') and env.current_solution.routes:
            for route in env.current_solution.routes:
                if not route: continue
                prev = 0  # depot is usually 0
                for node in route:
                    edges.add((min(prev, node), max(prev, node)))
                    prev = node
                edges.add((min(prev, 0), max(prev, 0))) # back to depot
        return frozenset(edges)

    def _get_cvrp_distance(self, edges_a, edges_b):
        """
        Calculates the symmetric difference between two CVRP edge-sets.
        Returns the absolute number of differing edges.
        """
        return len(edges_a.symmetric_difference(edges_b))

    def _add_to_local_pool(self, env: Env, current_best: float):
        """Add solution to elite pool with strict diversity check."""
        fingerprint = self._get_cvrp_fingerprint(env)
        
        # Check against existing to maintain diversity (Discard Hamming, use specialized CVRP edge diff)
        is_duplicate = False
        for elite in self.elite_pool:
            if abs(elite['value'] - current_best) < 1e-4:
                dist = self._get_cvrp_distance(elite['fingerprint'], fingerprint)
                if dist < 5: # Highly overlapping edges/structure
                    is_duplicate = True
                    break
        
        if is_duplicate:
            return False

        routes_copy = [list(r) for r in env.current_solution.routes] if hasattr(env, 'current_solution') else []
        
        elite_entry = {
            'value': current_best,
            'fingerprint': fingerprint,
            'routes': routes_copy,
            'timestamp': time.time()
        }
        
        self.elite_pool.append(elite_entry)
        self.elite_pool.sort(key=lambda x: x['value']) # Minimization problem
        
        if len(self.elite_pool) > self.POOL_CAPACITY:
            # Capacity reached: inheritance replacement (drop the worst)
            self.elite_pool = self.elite_pool[:self.POOL_CAPACITY]
            
        # Try to share this breakthrough
        self._save_to_shared_pool(elite_entry)
        return True

    def _save_to_shared_pool(self, entry):
        """Save a new breakthrough solution to the distributed filesystem mapping to shard."""
        if not self.shared_pool_dir: return
        
        pool_path = os.path.join(self.shared_pool_dir, f"pool_{self.pool_id}_{self.pool_type}")
        shard_path = os.path.join(pool_path, f"shard_{self.shard_id}")
        os.makedirs(shard_path, exist_ok=True)
        
        current_time = time.time()
        filename = f"sol_{entry['value']}_{int(current_time)}_{self.worker_id}_{random.randint(1000,9999)}.pkl"
        filepath = os.path.join(shard_path, filename)
        
        try:
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                pickle.dump(entry, f)
            os.rename(temp_path, filepath)
            
            # Validation Output
            self.logger(f"Saved elite solution to: {filepath}")
        except Exception as e:
            pass

    def _sync_shared_pool(self):
        """Syncs elite pool from file system."""
        if not self.shared_pool_dir: return
        
        self._check_and_update_pool_id()
        if self.pending_rebuild:
            return
            
        pool_path = os.path.join(self.shared_pool_dir, f"pool_{self.pool_id}_{self.pool_type}")
        if not os.path.exists(pool_path):
            return
            
        pkl_files = glob.glob(os.path.join(pool_path, "shard_*", "*.pkl"))
        if not pkl_files:
            return
            
        if len(pkl_files) > 50:
            pkl_files = random.sample(pkl_files, 50)
            
        for f in pkl_files:
            try:
                fname = os.path.basename(f)
                val_str = fname.split("_")[1]
                val = float(val_str)
                
                # Pre-filter by worst local elite
                if len(self.elite_pool) >= self.POOL_CAPACITY and val > self.elite_pool[-1]['value']:
                    continue
                    
                with open(f, 'rb') as fp:
                    entry = pickle.load(fp)
                    
                is_duplicate = False
                for elite in self.elite_pool:
                    if abs(elite['value'] - entry['value']) < 1e-4:
                        if self._get_cvrp_distance(elite['fingerprint'], entry['fingerprint']) < 5:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    self.elite_pool.append(entry)
            except Exception:
                continue
                
        self.elite_pool.sort(key=lambda x: x['value'])
        if len(self.elite_pool) > self.POOL_CAPACITY:
            self.elite_pool = self.elite_pool[:self.POOL_CAPACITY]

    def _run_epoch(self, env: Env) -> bool:
        """
        Single Worker's execution loop (Phase A -> B -> C -> D)
        """
        self.logger("Restarting with Constructive Heuristic...")
        
        # Fallback loop until solution is COMPLETE and VALID
        max_retries = 10
        for retry in range(max_retries):
            # --- Phase A: Cold Start (Initialization) ---
            env.clear_solution()
            
            construction_steps = 0
            prev_unvisited = 1000
            stagnation_counter = 0

            # CVRP construct loop until solution is complete (all nodes visited and legally routed)
            while not env.is_complete_solution and construction_steps < 1000:
                if not self.constructive_heuristics:
                    self.logger("Critical Failure: No constructive heuristics found.")
                    return False
                
                unvisited_count = len(env.problem_state.get("unvisited_nodes", []))
                
                # Check if we are stuck in a bin-packing local optimum
                if unvisited_count == prev_unvisited:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                prev_unvisited = unvisited_count

                # If stuck for 5 steps, inject a Ruin operator to make space!
                if stagnation_counter >= 5 and "mass_ruin" in self.breakout_heuristics:
                    ruin_candidates = self.breakout_heuristics["mass_ruin"]
                    if not isinstance(ruin_candidates, list):
                         ruin_candidates = [ruin_candidates]
                    h_ruin = random.choice(ruin_candidates)
                    try:
                        env.run_heuristic(h_ruin)
                    except Exception:
                        pass
                    stagnation_counter = 0 # reset after applying ruin
                else:
                    # Pick a random constructive heuristic
                    h = random.choice(self.constructive_heuristics)
                    try:
                        env.run_heuristic(h)
                    except Exception as e:
                        # Log silently or ignore to keep building
                        pass

                construction_steps += 1
            
            if env.is_complete_solution:
                self.logger(f"Construction completed. Value: {env.key_value}")
                break
            else:
                self.logger(f"Construction failed or incomplete (Value: {env.key_value}). Retrying ({retry+1}/{max_retries})...")
        
        if not env.is_complete_solution:
            self.logger("Critical Failure: Unable to construct valid solution after retries.")
            return False

        current_best = env.key_value
        no_improve_steps = 0
        self.current_run_steps = 0
        self.stagnation_level = 0
        self.phase_retries = 0
        self.last_restart_step = 0
        
        while env.continue_run:
            self.current_run_steps += 1
            
            # --- Phase B: Repair / Improve ---
            improved = self._run_improvement_phase(env)
            
            # [CRITICAL SECURITY CHECK] Prevent CVRP Invalid Route Exploit
            # If the breakout/repair failed to visit all nodes, the cost drops artificially to 170.
            # We must NEVER evaluate or accept this broken solution!
            if not env.is_complete_solution or not env.validation_solution():
                self.logger(f"Step:{self.current_run_steps} POISON DETECTED: Solution invalid! Forcing complete rebuild.")
                env.clear_solution()
                c_steps = 0
                while not env.is_complete_solution and c_steps < 1000:
                    if not self.constructive_heuristics: break
                    try: env.run_heuristic(__import__('random').choice(self.constructive_heuristics))
                    except: pass
                    c_steps += 1
                if not env.is_complete_solution:
                    return False # Abort epoch completely
            
            # --- Phase C: Check Status (Log Formats must align with Max-Cut) ---
            # Evaluate if a Breakthrough occurred
            # CVRP is a Min problem, so a lower cost is better
            
            if env.key_value < current_best - 1e-3: # Meaningful Cost decreased
                old_best = current_best
                current_best = env.key_value
                
                # Reset Stagnation & Climb Ladder
                no_improve_steps = 0
                self.stagnation_level = 0
                self.phase_retries = 0
                
                # Log state (aligned format)
                self.logger(f"Step:{self.current_run_steps} NEW LOCAL BEST: {current_best}")
                
                # Check Global Breakthrough
                if env.best_known is None or current_best < env.best_known - 1e-4:
                    self.logger(f"!!! BREAKTHROUGH: {old_best} > {current_best} !!!")
                    env.best_known = current_best
                    
                    # [SAFE SAVE STRATEGY]
                    saved_best = float('inf')
                    if env.output_dir and os.path.exists(env.output_dir):
                        for f in os.listdir(env.output_dir):
                            if f.startswith("breakthrough_") or f.startswith("match_"):
                                try:
                                    part = f.rsplit("_", 1)[-1] 
                                    score = float(part.replace(".txt", ""))
                                    if score < saved_best:
                                        saved_best = score
                                except: pass
                    
                    if (saved_best - current_best) > 1e-3:
                        env.dump_result(result_file=f"breakthrough_from_worker_{self.worker_id}_{current_best}.txt")

                elif env.best_known is not None and abs(current_best - env.best_known) <= 1e-4:
                     self.logger(f"~~~ MATCHED BEST KNOWN: {current_best} ~~~")
                     has_records = False
                     if env.output_dir and os.path.exists(env.output_dir):
                         for f in os.listdir(env.output_dir):
                             if f.startswith("breakthrough_") or f.startswith("match_"):
                                 has_records = True
                                 break
                     
                     if not has_records:
                        env.dump_result(result_file=f"match_best_known_from_worker_{self.worker_id}_{current_best}.txt")
                    
                # [Phase C: Elite Pool Sync] Add to Elite Pool
                self._add_to_local_pool(env, current_best)
                
            else:
                no_improve_steps += 1
                
                # Routine Log for Tracker Script
                if self.current_run_steps % 10 == 0:
                    self.logger(f"Step:{self.current_run_steps} Cur:{env.key_value} "
                                f"Best:{current_best} (BK:{env.best_known}) Stagnation:{no_improve_steps}")
            
            # --- Phase C (part 2): Periodic Sync ---
            if self.current_run_steps % 10 == 0:
                self._sync_shared_pool()
                if self.pending_rebuild:
                    # Leader called a rebuild L5, exit epoch to restart
                    return True

                
        return True
        
    def run(self, env: Env) -> bool:
        """Main entry point, wraps Event Epoch to handle L5 global hard restarts."""
        while True:
            result = self._run_epoch(env)
            
            if self.pending_rebuild:
                self.restart_count += 1
                if self.max_restarts is not None and self.restart_count > self.max_restarts:
                    self.logger(f" GLOBAL HARD RESTART TRIGGERED (Epoch {self.pool_id}) - ABORTING.")
                    return result
                
                self.logger(f" GLOBAL HARD RESTART TRIGGERED (Epoch {self.pool_id}) - Continue")
                
                # Reset Environment Logic for CVRP
                env.reset()
                self.elite_pool = []
                self.pending_rebuild = False
                continue
                
            return result
