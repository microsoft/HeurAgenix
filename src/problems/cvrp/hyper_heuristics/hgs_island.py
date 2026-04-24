import os
import time
import random
import glob
import pickle
import hashlib
from src.problems.cvrp.env import Env
from src.problems.cvrp.components import Solution, ReplaceSolutionOperator
from src.util.util import load_function

class HGSIslandHyperHeuristic:
    """
    CVRP-specialized HGS Island hyper-heuristic scheduling strategy (Skeleton version).
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
        
        # [PHASE 2 CONFIG - Aligned with Max-Cut]
        # Capacity limit for Global Epoch expansion (Total files on disk before inheritance).
        self.POOL_CAPACITY = 10000
        
        # [NEW] Elite Quality Filter Size (N)
        # For CVRP, we ONLY want to cross-over/relink with the absolute top tier, to prevent
        # transferring genes from terrible distant solutions.
        self.ELITE_FILTER_SIZE = 30
        
        # [NEW] Migration Count
        # Number of elites to migrate to new pool during inherit expansion
        self.MIGRATION_COUNT = 100

        # [NEW] Diversity Control
        # Minimum distance ratio for diversity checks (0.05 = 5% of nodes)
        self.MIN_DIST_RATIO_MIGRATION = 0.05 
        self.MIN_DIST_RATIO_RELINKING = 0.05
        
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
            "petal_algorithm_b384",
            "greedy_f4c4",
            "farthest_insertion_4e1d",
            "farthest_insertion_6308",
            "min_cost_insertion_048f",
            "min_cost_insertion_3b2b",
            "random_bfdc",
            "regret_insertion_2f3a", # Include strong operator for tight capacities
            "first_fit_decreasing_bfd"
        }
        
        improvement_names = {
            "saving_algorithm_710e",
            "two_opt_0554",
            "three_opt_e8d7",
            "node_shift_between_routes_7b8a",
            "variable_neighborhood_search_614b",
            "giant_tour_dp_split",
            "swap_star",
            "or_opt_segment_relocate"
        }
        
        breakout_map = {
            "mass_ruin": ["radial_ruin_3c4d", "random_ruin_1a2b", "sisr_ruin", "sisr_ruin_4a5b"],
            "recreate": ["regret_insertion_2f3a", "min_cost_insertion_048f", "min_cost_insertion_3b2b"],
            "crossover": ["route_based_crossover_9f8a", "hgs_giant_tour_crossover"]
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
                    if key not in self.breakout_heuristics:
                        self.breakout_heuristics[key] = []
                    self.breakout_heuristics[key].append(func)
        
    def _get_pool_path(self, pool_id, pool_type):
        """Returns the directory path for a specific Epoch Pool."""
        if not self.shared_pool_dir: return ""
        return os.path.join(self.shared_pool_dir, f"pool_{pool_id}_{pool_type}")
        
    def _get_shard_path(self, bucket_path, shard_index):
        return os.path.join(bucket_path, f"shard_{shard_index}")

    # =====================================================================
    # 3. Core Solving Phases Strategy
    # =====================================================================
    def _run_improvement_phase(self, env, time_limit=30.0):
        """
        Phase B: VND (Variable Neighborhood Descent) with time safety valve.
        Continuously apply improvement operators until local optimum or time limit.
        """
        if not self.improvement_heuristics: return False
        
        max_vnd_loops = 500  # Hard cap per VND call
        total_improved = False
        start_time = time.time()
        
        heuristics_queue = list(self.improvement_heuristics)
        
        for loop_idx in range(max_vnd_loops):
            # Time safety valve
            if time.time() - start_time > time_limit:
                break
                
            improved_in_this_loop = False
            random.shuffle(heuristics_queue)
            
            for heuristic in heuristics_queue:
                if time.time() - start_time > time_limit:
                    break
                    
                # 1. Snapshot State
                backup_wrapper = env.export_solution_wrapper()
                start_val = backup_wrapper["solution"].total_cost
                
                # 2. Run Heuristic
                try:
                    env.run_heuristic(heuristic)
                except Exception as e:
                    env.import_solution_wrapper(backup_wrapper)
                    continue

                # 3. Acceptance Criteria: Strict Descent (CVRP is Min Problem)
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
                # Primary sort by id, secondary fallback to give 'rebuild' priority if same ID (shouldn't happen)
                pools.sort(key=lambda x: (x[0], 1 if x[1] == 'rebuild' else 0))
                return pools[-1]
            return (self.pool_id, self.pool_type)
        except Exception as e:
            self.logger(f"Error scanning pool epochs: {e}")
            return (self.pool_id, self.pool_type)

    def _check_and_update_pool_id(self):
        """Scans periodically and updates the pool pointer if a new epoch is found."""
        old_id = self.pool_id
        new_id, new_type = self._scan_pool_epochs()
        
        # Priority to Rebuild: Follow if it's a completely new ID, OR if it's the same ID but a 'rebuild' taking over an 'inherit'
        if new_id > old_id or (new_id == old_id and new_type == 'rebuild' and self.pool_type != 'rebuild'):
            self.pool_id = new_id
            self.pool_type = new_type
            self.logger(f"-> [FOLLOW] Detected REVOLUTION (Pool {new_id}_{new_type}). Resetting...")
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

    def _get_pure_distance_cost(self, env: Env) -> float:
        """Calculate pure distance cost WITHOUT capacity penalty (for fair Elite Pool comparison)."""
        total = 0.0
        for route in env.current_solution.routes:
            if not route: continue
            total += env._get_route_cost(route)
        return total

    def _is_feasible(self, env: Env) -> bool:
        """Check if current solution respects all capacity constraints."""
        demands = env.instance_data['demands']
        capacity = env.instance_data['capacity']
        return all(
            sum(demands[n] for n in route) <= capacity
            for route in env.current_solution.routes
        )

    def _add_to_local_pool(self, env: Env, current_best: float):
        """Add solution to elite pool with strict diversity check. Only stores feasible solutions with pure distance cost."""
        # HGS: Only add feasible solutions to elite pool
        if not self._is_feasible(env):
            return False
            
        pure_cost = self._get_pure_distance_cost(env)
        fingerprint = self._get_cvrp_fingerprint(env)
        
        # Check against existing to maintain diversity
        is_duplicate = False
        for elite in self.elite_pool:
            if abs(elite['value'] - pure_cost) < 1e-4:
                dist = self._get_cvrp_distance(elite['fingerprint'], fingerprint)
                if dist < 5: # Highly overlapping edges/structure
                    is_duplicate = True
                    break
        
        if is_duplicate:
            return False

        routes_copy = [list(r) for r in env.current_solution.routes] if hasattr(env, 'current_solution') else []
        
        elite_entry = {
            'value': pure_cost,
            'fingerprint': fingerprint,
            'routes': routes_copy,
            'timestamp': time.time()
        }
        
        self.elite_pool.append(elite_entry)
        self.elite_pool.sort(key=lambda x: x['value']) # Minimization problem
        
        # HGS-style diversity-aware survivor selection
        self._hgs_survivor_selection()
        
        if len(self.elite_pool) > self.POOL_CAPACITY:
            # Capacity reached: inheritance replacement (drop the worst)
            self.elite_pool = self.elite_pool[:self.POOL_CAPACITY]
            
        # Try to share this breakthrough
        self._save_to_shared_pool(elite_entry)
        return True

    def _save_to_shared_pool(self, entry):
        """Save a new breakthrough solution to the distributed filesystem mapping to shard."""
        if not self.shared_pool_dir: return
        
        # Double check we haven't been forcefully obsoleted by a Rebuild before writing
        new_id, new_type = self._scan_pool_epochs()
        if new_id > self.pool_id:
            # We are holding a ghost solution from a past epoch, let the check_and_update grab it later
            return
        
        if new_id == self.pool_id and new_type == 'rebuild' and self.pool_type != 'rebuild':
            # DO NOT write legacy inherit solutions into a fresh rebuild pool
            return
            
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

    def _apply_breakout(self, env: Env, strategy: str):
        # [DYNAMIC PENALTY LADDER START] Sharp penalty drop to cross infeasible valley!
        if strategy in ["elite_route_injection", "macro_route_ruin", "targeted_ruin"]:
            env.problem_state["capacity_penalty_factor"] = 2.0
            self._penalty_ladder_active = True

        if strategy == "targeted_ruin":
            # [L1 - Targeted Small Ruin & Recreate]
            # Precise 5%-15% geographic/cost ruin followed by regret insertion.
            ratio = random.uniform(0.05, 0.15)
            
            # Select operators
            ruin_h = random.choice(self.breakout_heuristics["mass_ruin"]) if "mass_ruin" in self.breakout_heuristics else None
            recreate_h = random.choice(self.breakout_heuristics["recreate"]) if "recreate" in self.breakout_heuristics else None
            
            if not ruin_h or not recreate_h:
                self.logger("Warning: Missing required operators for L1 breakout.")
                return
                
            # 1. Take snapshot for safe rollback
            backup_wrapper = env.export_solution_wrapper()
            
            # 2. Execute Ruin (becomes incomplete)
            try:
                env.run_heuristic(ruin_h, parameters={"removal_fraction": ratio})
            except Exception as e:
                self.logger(f"Ruin failed: {e}")
                env.import_solution_wrapper(backup_wrapper)
                return
                
            # 3. Execute Recreate loop (restores feasibility)
            c_steps = 0
            while not env.is_complete_solution and c_steps < 100:
                try:
                    op = env.run_heuristic(recreate_h)
                    if not op or isinstance(op, str):
                        break  # Early breakout if operator fails or does nothing saving CPU
                except Exception as e:
                    self.logger(f"Recreate step failed: {e}")
                    break
                c_steps += 1
                
            if not env.is_complete_solution:
                self.logger("Recreate could not complete solution. Rolling back.")
                env.import_solution_wrapper(backup_wrapper)
                return
                
        elif strategy == "elite_route_injection":
            # [L2 - Elite Route Injection / Crossover]
            if len(self.elite_pool) < 2:
                self.logger("Elite pool too small for Crossover. Falling back to L1 targeted_ruin.")
                return self._apply_breakout(env, "targeted_ruin")
            
            # Strategy mix: 50% crossover, 50% direct elite import + perturbation
            use_direct_import = (random.random() < 0.5)
            
            # Find best elite that is different from current solution
            fingerprint = self._get_cvrp_fingerprint(env)
            candidate_pool = self.elite_pool[:self.ELITE_FILTER_SIZE] if hasattr(self, 'ELITE_FILTER_SIZE') else self.elite_pool
            
            candidates = []
            for elite in candidate_pool:
                dist = self._get_cvrp_distance(fingerprint, elite['fingerprint'])
                if dist >= 3:  # Lower threshold to accept more candidates
                    candidates.append((dist, elite))
            
            if not candidates:
                self.logger("Active Relinking: All elites too similar. Falling back to L1.")
                return self._apply_breakout(env, "targeted_ruin")
            
            if use_direct_import:
                # === DIRECT ELITE IMPORT === 
                # Pick the BEST quality elite (not most distant) and start VND from there
                # This is the key HGS mechanism: educate offspring from best parents
                best_candidates = sorted(candidates, key=lambda x: x[1]['value'])[:5]
                _, target_elite = random.choice(best_candidates)
                
                backup_wrapper = env.export_solution_wrapper()
                
                op = ReplaceSolutionOperator(routes=[list(r) for r in target_elite['routes']])
                env.run_operator(op)
                
                # Apply small perturbation (5-10% ruin) so VND can find new improving moves
                ratio = random.uniform(0.05, 0.10)
                ruin_h = random.choice(self.breakout_heuristics["mass_ruin"]) if "mass_ruin" in self.breakout_heuristics else None
                recreate_h = random.choice(self.breakout_heuristics["recreate"]) if "recreate" in self.breakout_heuristics else None
                
                if ruin_h and recreate_h:
                    try:
                        env.run_heuristic(ruin_h, parameters={"removal_fraction": ratio})
                    except Exception:
                        env.import_solution_wrapper(backup_wrapper)
                        return
                    
                    c_steps = 0
                    while not env.is_complete_solution and c_steps < 100:
                        try:
                            op = env.run_heuristic(recreate_h)
                            if not op or isinstance(op, str): break
                        except Exception: break
                        c_steps += 1
                    
                    if not env.is_complete_solution:
                        env.import_solution_wrapper(backup_wrapper)
                        return
            else:
                # === CROSSOVER INJECTION (original L2) ===
                candidates.sort(key=lambda x: x[0], reverse=True)
                top_candidates = candidates[:min(3, len(candidates))]
                chosen_dist, target_elite_dict = random.choice(top_candidates)
                
                # Construct Solution object for target
                target_sol = Solution(
                    routes=target_elite_dict['routes'],
                    depot=env.problem_state.get('depot', 0),
                    total_cost=target_elite_dict['value']
                )
                
                # Select operators
                crossover_h = random.choice(self.breakout_heuristics["crossover"]) if "crossover" in self.breakout_heuristics else None
                recreate_h = random.choice(self.breakout_heuristics["recreate"]) if "recreate" in self.breakout_heuristics else None
                
                if not crossover_h or not recreate_h:
                    self.logger("Warning: Missing required operators for L2 breakout.")
                    return self._apply_breakout(env, "targeted_ruin")
                    
                # 0. Take snapshot for safe rollback
                backup_wrapper = env.export_solution_wrapper()
                    
                # 1. Execute Crossover (Inject Elite Route -> invalidates overlaps, creates unfilled nodes)
                try:
                    env.run_heuristic(crossover_h, parameters={"target_solution": target_sol})
                except Exception as e:
                    self.logger(f"Crossover injection failed: {e}")
                    env.import_solution_wrapper(backup_wrapper)
                    return self._apply_breakout(env, "targeted_ruin")
                    
                # 2. Execute Recreate loop to insert the unassigned nodes 
                c_steps = 0
                while not env.is_complete_solution and c_steps < 100:
                    try:
                        op = env.run_heuristic(recreate_h)
                        if not op or isinstance(op, str):
                            break
                    except Exception as e:
                        self.logger(f"Recreate step failed after crossover: {e}")
                        break
                    c_steps += 1
                    
                if not env.is_complete_solution:
                    self.logger("Recreate could not complete solution after crossover. Rolling back.")
                    env.import_solution_wrapper(backup_wrapper)
                    return
                
        elif strategy == "macro_route_ruin":
            # [L3 - Macro / Route-Ejection Ruin]
            # Execute large-scale random destruction (30%-40%) to force a macro topology change.
            # Using mass_ruin (random or radial) with a much larger parameter to simulate destroying 2-3 entire routes.
            ratio = random.uniform(0.30, 0.40)
            ruin_h = random.choice(self.breakout_heuristics["mass_ruin"]) if "mass_ruin" in self.breakout_heuristics else None
            recreate_h = random.choice(self.breakout_heuristics["recreate"]) if "recreate" in self.breakout_heuristics else None
            
            if not ruin_h or not recreate_h:
                self.logger("Warning: Missing required operators for L3 breakout.")
                return
                
            # 1. Take snapshot for safe rollback
            backup_wrapper = env.export_solution_wrapper()
                
            try:
                env.run_heuristic(ruin_h, parameters={"removal_fraction": ratio})
            except Exception as e:
                self.logger(f"Macro Ruin failed: {e}")
                env.import_solution_wrapper(backup_wrapper)
                return
                
            # 2. Execute Recreate loop
            c_steps = 0
            while not env.is_complete_solution and c_steps < 100:
                try:
                    op = env.run_heuristic(recreate_h)
                    if not op or isinstance(op, str):
                        break
                except Exception as e:
                    self.logger(f"Macro Recreate step failed: {e}")
                    break
                c_steps += 1
                
            if not env.is_complete_solution:
                self.logger("Macro Recreate could not complete solution. Rolling back.")
                env.import_solution_wrapper(backup_wrapper)
                return
                
        elif strategy == "soft_restart":
             self.logger("... Soft Restart Triggered ... Abandoning current solution.")
             
             force_constructive = False
             
             if self.elite_pool and len(self.elite_pool) > 5:
                  # 1. Option A: Jump to a Distant Elite
                  # Calculate distance to current
                  current_fp = self._get_cvrp_fingerprint(env)
                  
                  # Find furthest elites within top filtered quality
                  candidates = []
                  candidate_pool = self.elite_pool[:self.ELITE_FILTER_SIZE] if hasattr(self, 'ELITE_FILTER_SIZE') else self.elite_pool
                  for elite in candidate_pool:
                      dist = self._get_cvrp_distance(current_fp, elite['fingerprint'])
                      candidates.append((dist, elite))
                  
                  candidates.sort(key=lambda x: x[0], reverse=True)
                  # Pick from top 5 furthest
                  target_dist, target_elite = random.choice(candidates[:5])
                  
                  if target_dist < max(10, int(len(env.instance_data.get('demands', [])) * 0.1)):
                       self.logger(f"Soft Restart Aborted: Pool Homogenized (Max Dist={target_dist}). Forcing construction.")
                       force_constructive = True
                  else:
                       # 2. Re-import selected Elite into Environment
                       op = ReplaceSolutionOperator(routes=[list(r) for r in target_elite['routes']])
                       env.run_operator(op)
                       self.logger(f"Restarted from Distant Elite (Val: {target_elite['value']}, Dist: {target_dist})")
             else:
                  force_constructive = True
                  
             if force_constructive:
                  # 1. Option B: Cold Build
                  self.logger("Restarting with Constructive Heuristic (Cold Build)...")
                  env.clear_solution()
                  c_steps = 0
                  while not env.is_complete_solution and c_steps < 1000:
                      if not self.constructive_heuristics: break
                      env.run_heuristic(random.choice(self.constructive_heuristics))
                      c_steps += 1
                  if not env.is_complete_solution:
                      self.logger("Soft restart construction failed.")
                 
        else:
            self.logger(f"Warning: Unknown breakout strategy '{strategy}'.")

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
            
        # Parse the values from the filenames to ensure we always get the *best* items
        # Format: sol_2157.0_1775402373_14_5644.pkl
        file_entries = []
        for f in pkl_files:
            try:
                fname = os.path.basename(f)
                val = float(fname.split("_")[1])
                file_entries.append((val, f))
            except Exception:
                continue
                
        file_entries.sort(key=lambda x: x[0])  # Sort by value ascending (since CVRP is minimization)
        
        # Extract the absolute top 50 global elites across all workers
        top_files = [f for v, f in file_entries[:50]]
            
        for f in top_files:
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

    # =====================================================================
    # HGS-Inspired: Dynamic Penalty Factor Adaptation
    # =====================================================================
    def _adapt_penalty_factor(self, env: Env):
        """
        HGS core mechanism: Dynamically adjust the capacity violation penalty factor
        based on the ratio of feasible solutions encountered recently.
        
        - If too many feasible solutions (>= target_feasible_ratio): DECREASE penalty to encourage
          exploring infeasible space (can find shortcuts).
        - If too few feasible solutions (< target_feasible_ratio): INCREASE penalty to push
          search back towards feasibility.
        
        Reference: Vidal et al. (2012) "A hybrid genetic search for the CVRP"
        """
        target_feasible_ratio = 0.25  # HGS default: aim for ~25% feasible
        adapt_factor = 1.2  # Moderate adjustment (HGS uses 1.2)
        min_penalty = 50.0   # Don't go too low — prevents infeasible cost < feasible cost
        max_penalty = 5000.0
        
        # Check current solution feasibility
        is_feasible = all(
            sum(env.instance_data['demands'][n] for n in route) <= env.instance_data['capacity']
            for route in env.current_solution.routes
        )
        
        # Track recent feasibility history 
        if not hasattr(self, '_feasibility_history'):
            self._feasibility_history = []
        self._feasibility_history.append(1 if is_feasible else 0)
        
        # Keep a sliding window of 50 recent solutions
        window = 50
        if len(self._feasibility_history) > window:
            self._feasibility_history = self._feasibility_history[-window:]
        
        # Only adapt after enough observations
        if len(self._feasibility_history) < 10:
            return
            
        current_ratio = sum(self._feasibility_history) / len(self._feasibility_history)
        current_pf = getattr(env, 'penalty_factor', 200.0)
        
        if current_ratio > target_feasible_ratio + 0.05:
            # Too many feasible → decrease penalty to explore infeasible space
            new_pf = max(min_penalty, current_pf / adapt_factor)
        elif current_ratio < target_feasible_ratio - 0.05:
            # Too few feasible → increase penalty to push back
            new_pf = min(max_penalty, current_pf * adapt_factor)
        else:
            return  # Within target range, no adjustment needed
        
        env.penalty_factor = new_pf
        env.problem_state["capacity_penalty_factor"] = new_pf

    # =====================================================================
    # HGS-Inspired: Survivor Selection with Diversity
    # =====================================================================
    def _hgs_survivor_selection(self):
        """
        When elite pool is at capacity, remove the solution that contributes least
        to diversity (smallest average distance to neighbors), breaking ties by quality.
        This prevents the pool from collapsing to a cluster of near-identical solutions.
        """
        if len(self.elite_pool) <= self.ELITE_FILTER_SIZE:
            return
            
        # Calculate contribution score for each: avg distance to closest 3 neighbors
        scores = []
        for i, sol in enumerate(self.elite_pool):
            dists = []
            for j, other in enumerate(self.elite_pool):
                if i == j: continue
                d = self._get_cvrp_distance(sol['fingerprint'], other['fingerprint'])
                dists.append(d)
            dists.sort()
            # Average of 3 closest neighbors
            close_avg = sum(dists[:3]) / max(1, min(3, len(dists)))
            scores.append((i, close_avg, sol['value']))
        
        # Find worst contributor: lowest diversity score, breaking ties by worst quality
        scores.sort(key=lambda x: (x[1], -x[2]))  # ascending diversity, descending cost
        worst_idx = scores[0][0]
        
        # Only remove if pool is over capacity
        if len(self.elite_pool) > self.ELITE_FILTER_SIZE:
            del self.elite_pool[worst_idx]

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
        best_wrapper = env.export_solution_wrapper()
        no_improve_steps = 0
        self.current_run_steps = 0
        self.stagnation_level = 0
        self.phase_retries = 0
        self.last_restart_step = 0
        
        while env.continue_run:
            self.current_run_steps += 1
            
            # --- HGS: Adapt penalty factor before improvement ---
            self._adapt_penalty_factor(env)
            
            # --- Phase B: Repair / Improve ---
            improved = self._run_improvement_phase(env)
            
            # [CRITICAL SECURITY CHECK] Prevent CVRP Invalid Route Exploit
            # If the breakout/repair failed to visit all nodes, the cost drops artificially to 170.
            # We must NEVER evaluate or accept this broken solution!
            if not env.is_complete_solution or not env.validation_solution():
                self.logger(f"Step:{self.current_run_steps} POISON DETECTED: Solution invalid! Forcing rollback to best.")
                env.import_solution_wrapper(best_wrapper)
                # It continues as the old best, skipping Phase C update as it's == current_best
            
            # --- Phase C: Check Status (Log Formats must align with Max-Cut) ---
            # Evaluate if a Breakthrough occurred
            # CVRP is a Min problem, so a lower cost is better
            
            if env.key_value < current_best - 1e-3: # Meaningful Cost decreased
                old_best = current_best
                current_best = env.key_value
                best_wrapper = env.export_solution_wrapper()
                
                is_sol_feas = self._is_feasible(env)
                pure_cost = self._get_pure_distance_cost(env) if is_sol_feas else None
                
                if is_sol_feas and (not hasattr(self, 'global_best_cost') or pure_cost < self.global_best_cost):
                    self.global_best_cost = pure_cost
                    self.global_best_wrapper = env.export_solution_wrapper()
                
                # Reset Stagnation — but preserve level if improvement came from L2+
                # This prevents the "L2 success → reset to L0 → slow L1 grind → L2 again" cycle
                no_improve_steps = 0
                if self.stagnation_level >= 2:
                    # Keep at L1 so we can quickly escalate back to L2 crossover
                    self.stagnation_level = 1
                    self.phase_retries = 0
                else:
                    self.stagnation_level = 0
                    self.phase_retries = 0
                
                # Log state (aligned format)
                feas_tag = "" if is_sol_feas else " [INFEASIBLE]"
                cost_display = f"{pure_cost:.0f}" if is_sol_feas else f"{env.key_value:.0f}*"
                self.logger(f"Step:{self.current_run_steps} NEW LOCAL BEST: {cost_display}{feas_tag}")
                
                # Check Global Breakthrough — ONLY for feasible solutions using pure distance
                # Aligned with max_cut style: breakthrough updates env.best_known, match only recorded once
                if is_sol_feas and pure_cost is not None:
                    feasible_bk = getattr(self, '_feasible_best_known', float('inf'))
                    if pure_cost < feasible_bk - 1e-4:
                        self._feasible_best_known = pure_cost
                    
                    # CVRP is Min: breakthrough = beat (lower than) env.best_known
                    if pure_cost < env.best_known - 1e-4:
                        self.logger(f"!!! BREAKTHROUGH: {pure_cost:.0f} < {env.best_known:.0f} !!!")
                        env.best_known = pure_cost  # Update so only further improvements trigger again
                        
                        # [SAFE SAVE] Only save if strictly better than anything on disk
                        saved_best = float('inf')
                        if os.path.exists(env.output_dir):
                            for f in os.listdir(env.output_dir):
                                if f.startswith("breakthrough_") or f.startswith("match_"):
                                    try:
                                        part = f.rsplit("_", 1)[-1]
                                        score = float(part.replace(".txt", ""))
                                        if score < saved_best:
                                            saved_best = score
                                    except: pass
                        if pure_cost < saved_best - 1e-3:
                            env.dump_result(result_file=f"breakthrough_from_worker_{self.worker_id}_{pure_cost:.0f}.txt")
                    
                    elif abs(pure_cost - env.best_known) <= 1e-4:
                        self.logger(f"~~~ MATCHED BEST KNOWN: {pure_cost:.0f} ~~~")
                        # Only save Match if no results exist yet (first match only)
                        has_records = False
                        if os.path.exists(env.output_dir):
                            for f in os.listdir(env.output_dir):
                                if f.startswith("breakthrough_") or f.startswith("match_"):
                                    has_records = True
                                    break
                        if not has_records:
                            env.dump_result(result_file=f"match_best_known_from_worker_{self.worker_id}_{pure_cost:.0f}.txt")
                    
                # [Phase C: Elite Pool Sync] Add to Elite Pool (only feasible)
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

            # --- Phase D: Breakout / Ruin Strategies ---
            # For CVRP, since we run a full VND loop in Phase B that exhausts all improvement moves, 
            # we reach a local optimum almost instantly. 
            # Thus, patience should be zero to avoid wasting cycles checking an already converged solution.
            node_num = env.instance_data.get("node_num", 80) if hasattr(env, 'instance_data') else 80
            patience = 0
            
            if no_improve_steps > patience:
                # [Dynamic Retry Setting] Faster escalation to L2+ where elite pool is leveraged
                max_retries_per_phase = max(12, int(node_num * 0.2))
                self.phase_retries += 1
                
                # Check relation to Elite Pool (Global Best)
                is_attacking_global_best = False
                global_best_val = float('inf')
                if self.elite_pool:
                     global_best_val = min(s["value"] for s in self.elite_pool)
                     # Using tolerance for CVRP (minimization)
                     if current_best <= global_best_val + 1e-3:
                         is_attacking_global_best = True

                if self.stagnation_level == 0:
                    self.stagnation_level = 1
                    self.phase_retries = 1
                    strategy = "targeted_ruin"
                elif self.stagnation_level == 1 and self.phase_retries <= max_retries_per_phase:
                    strategy = "targeted_ruin"
                elif self.stagnation_level == 1 and self.phase_retries > max_retries_per_phase:
                    # Upgrade to L2
                    self.stagnation_level = 2
                    self.phase_retries = 1
                    strategy = "elite_route_injection"
                    self.logger(f"Escalating Stagnation Level to {self.stagnation_level} (Exhausted L1 retries)")
                elif self.stagnation_level == 2 and self.phase_retries <= max_retries_per_phase:
                    strategy = "elite_route_injection"
                elif self.stagnation_level == 2 and self.phase_retries > max_retries_per_phase:
                    # Upgrade to L3
                    self.stagnation_level = 3
                    self.phase_retries = 1
                    strategy = "macro_route_ruin"
                    self.logger(f"Escalating Stagnation Level to {self.stagnation_level} (Exhausted L2 retries)")
                elif self.stagnation_level == 3 and self.phase_retries <= max_retries_per_phase:
                    strategy = "macro_route_ruin"
                elif self.stagnation_level == 3 and self.phase_retries > max_retries_per_phase:
                    # Upgrade to L4
                    self.stagnation_level = 4
                    self.phase_retries = 1
                    strategy = "soft_restart"
                    self.logger(f"Escalating Stagnation Level to {self.stagnation_level} (Exhausted L3 retries)")
                elif self.stagnation_level == 4 and self.phase_retries <= max_retries_per_phase:
                    strategy = "soft_restart"
                else:
                    self.stagnation_level += 1
                    self.phase_retries = 1
                    self.logger(f"Escalating Stagnation Level to {self.stagnation_level} (Exhausted L4 retries)")
                    strategy = "soft_restart"
                    
                # Evaluate Hard Restarts explicitly
                if self.stagnation_level >= 5:
                    if is_attacking_global_best:
                        self.logger(f"Step:{self.current_run_steps} Exhausted L4 ({max_retries_per_phase} retries). Triggering L5 (Global Rebuild).")
                        
                        # 1. Intent check and preemptive creation (Concurrency Control)
                        if self.shared_pool_dir:
                            if self.max_restarts is not None and self.restart_count >= self.max_restarts:
                                self.logger(f"Max restarts ({self.max_restarts}) reached. Skipping pool creation.")
                                self.pending_rebuild = True
                                return True
                                
                            new_pool_id = self.pool_id + 1
                            new_pool_type = 'rebuild'
                            pool_path = os.path.join(self.shared_pool_dir, f"pool_{new_pool_id}_{new_pool_type}")
                            try:
                                os.makedirs(pool_path, exist_ok=False)  # Atomic creation
                                self.pool_id = new_pool_id
                                self.pool_type = new_pool_type
                                self.logger(f"Initiated NEW REBUILD Epoch: {new_pool_id}_{new_pool_type}")
                            except FileExistsError:
                                self.logger(f"Conflict: Rebuild Epoch {new_pool_id}_{new_pool_type} already created by another worker. Obeying.")
                                self.pool_id = new_pool_id
                                self.pool_type = new_pool_type
                        
                        self.pending_rebuild = True
                        return True
                    else:
                        self.logger(f"Step:{self.current_run_steps} L5 Detected, but Local Best ({current_best}) > Global Best ({global_best_val}). Downgrading to L4 Soft Restart.")
                        strategy = "soft_restart"
                        self.stagnation_level = 4
                        self.phase_retries = 0

                # [FIX]: Revert to the tracking best solution to ensure we are always perturbing our peak, 
                # instead of drifting into a random walk sequence of failed breakouts.
                if strategy != "soft_restart" and env.key_value > current_best * 1.15:
                    env.import_solution_wrapper(best_wrapper)

                self.logger(f"Step:{self.current_run_steps} Stagnation L{self.stagnation_level} (Try {self.phase_retries}/{max_retries_per_phase}). Qual={env.key_value:.0f} Act={strategy}")
                
                # Apply Breakout (Ruin & Recreate) before going back to Improve phase
                self._apply_breakout(env, strategy)
                
                # If L4 triggers, worker abandons trajectory. We MUST reset current_best to track the new trajectory.
                if strategy == "soft_restart":
                    if is_attacking_global_best:
                        self.logger("Leader Mode: Retaining high 'current_best' baseline to force meaningful improvement across Soft Restarts.")
                    else:
                        self.logger("Follower Mode: Resetting 'current_best' to allow local hill climbing.")
                        current_best = env.key_value
                        best_wrapper = env.export_solution_wrapper()
                        self.stagnation_level = 0
                        self.phase_retries = 0
                
                # Reset counter to give the new candidate a chance
                no_improve_steps = 0

                
        return True
        
    def run(self, env: Env) -> bool:
        """Main entry point, wraps Event Epoch to handle L5 global hard restarts."""
        while True:
            result = self._run_epoch(env)
            
            if getattr(self, 'pending_rebuild', False):
                self.restart_count += 1
                if self.max_restarts is not None and self.restart_count > self.max_restarts:
                    self.logger(f" GLOBAL HARD RESTART TRIGGERED (Epoch {self.pool_id}) - ABORTING.")
                    return result
                
                self.logger(f" GLOBAL HARD RESTART TRIGGERED (Epoch {self.pool_id}_{self.pool_type}) - Continue")
                
                # [L5 CORE: Reset Environment Logic for CVRP - Scorched Earth]
                env.reset()
                
                # pool_id already incremented in Phase D breakout block using Atomic Filesys Create
                
                self.elite_pool = [] # Destroy all locally accumulated Elite solutions
                
                # [CRITICAL FIX]: Immediately seed the new epoch with our absolute global best
                if hasattr(self, 'global_best_wrapper') and getattr(self, 'global_best_wrapper') is not None:
                    # Temporarily load it into env to get the fingerprint and add it
                    env.import_solution_wrapper(self.global_best_wrapper)
                    self._add_to_local_pool(env, self.global_best_cost)
                    env.reset() # Re-reset env context to actually restart
                    
                self.pending_rebuild = False
                continue
                
            return result
