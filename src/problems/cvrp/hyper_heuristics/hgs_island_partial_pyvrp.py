import os
import time
import random
import glob
import pickle
import hashlib
from src.problems.cvrp.env import Env

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
        # Hybrid mode: use a short PyVRP run as a seed, then continue with our framework.
        # 屏蔽pyVRP相关参数
        self.enable_pyvrp_seed = False
        self.pyvrp_seed_runtime = 0
        self.pyvrp_seed_attempts = 0
        self.pyvrp_reseed_runtime = 0
        self.pyvrp_reseed_runtime_max = 0
        self.pyvrp_reseed_attempts = 0
        self.pyvrp_reseed_cooldown_steps = 0
        self.last_pyvrp_reseed_step = -10**9
        # Optional partial use: only for near-BK micro polish, not full initialization.
        self.enable_micro_pyvrp_polish = bool(kwargs.get("enable_micro_pyvrp_polish", True))
        self.micro_pyvrp_runtime = int(kwargs.get("micro_pyvrp_runtime", 120))
        self.micro_pyvrp_attempts = int(kwargs.get("micro_pyvrp_attempts", 4))
        self.micro_pyvrp_max_calls = int(kwargs.get("micro_pyvrp_max_calls", 24))
        self.micro_pyvrp_calls = 0
        self.restart_count = 0
        
        # [NEW] Mixed Initialization Strategy: Some workers use pure construction, some use PyVRP
        # This proves our HGS framework has independent merit beyond PyVRP dependency.
        self.enable_construction_baseline = kwargs.get("enable_construction_baseline", True)
        self.construction_init_ratio = float(kwargs.get("construction_init_ratio", 0.3))  # 30% use pure construction
        # Decide for this worker on init (once per worker instance)
        self.use_construction_init = (random.random() < self.construction_init_ratio)
        
        # Heuristics lists
        self.constructive_heuristics = []
        self.fast_improvement_heuristics = []
        self.heavy_improvement_heuristics = []
        self.improvement_heuristics = []
        self.ruin_heuristics = []
        self.breakout_heuristics = {}
        
        self._classify_heuristics()
        
        # State tracking
        self.elite_pool = []
        # PyVRP-style dual sub-populations: keep both feasible and infeasible
        # candidates so crossover can exploit boundary solutions.
        self.feasible_subpop = []
        self.infeasible_subpop = []
        self.FEASIBLE_SUBPOP_MAX = 36
        self.INFEASIBLE_SUBPOP_MAX = 24
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
        self.REBUILD_SEED_COUNT = 4
        
        # Initialize Base Pool Directory
        if self.shared_pool_dir:
            try:
                self.logger(f"Shared Elite Pool Directory: {self.shared_pool_dir}")
                os.makedirs(self.shared_pool_dir, exist_ok=True)
                # TODO: Check for existing epoch pools and sync state
            except OSError:
                pass 

    def _try_pyvrp_warm_start(self, env: Env, runtime: int | None = None, seed: int | None = None, attempts: int = 1) -> bool:
        # 已屏蔽pyVRP，直接返回False
        return False

    def _try_micro_pyvrp_polish(self, env: Env, current_best: float | None = None) -> bool:
        """Use a tiny PyVRP call as a late-stage polisher near BK (limited times)."""
        if not self.enable_micro_pyvrp_polish:
            return False
        if self.micro_pyvrp_calls >= self.micro_pyvrp_max_calls:
            return False
        pyvrp_heuristic = self.breakout_heuristics.get("pyvrp_blackbox")
        if pyvrp_heuristic is None:
            return False
        if not self._is_feasible(env):
            return False

        if current_best is None:
            current_best = self._get_pure_distance_cost(env)

        backup_wrapper = env.export_solution_wrapper()
        try:
            env.run_heuristic(
                pyvrp_heuristic,
                parameters={
                    "data_path": getattr(env, "data_path", ""),
                    "runtime": max(1, self.micro_pyvrp_runtime),
                    "attempts": max(1, self.micro_pyvrp_attempts),
                    "seed": None,
                    "worker_id": self.worker_id,
                }
            )
            if not env.is_complete_solution or not env.validation_solution() or (not self._is_feasible(env)):
                env.import_solution_wrapper(backup_wrapper)
                return False

            new_cost = self._get_pure_distance_cost(env)
            if new_cost < current_best - 1e-3:
                self.micro_pyvrp_calls += 1
                self.last_pyvrp_reseed_step = self.current_run_steps
                self.logger(
                    f"[MicroPyVRP] accepted polish: {current_best:.0f} -> {new_cost:.0f} "
                    f"(runtime={self.micro_pyvrp_runtime}s, calls={self.micro_pyvrp_calls}/{self.micro_pyvrp_max_calls})"
                )
                return True

            env.import_solution_wrapper(backup_wrapper)
            return False
        except Exception as e:
            env.import_solution_wrapper(backup_wrapper)
            self.logger(f"[MicroPyVRP] polish failed: {e}")
            return False

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
            "first_fit_decreasing_bfd",
        }
        
        fast_improvement_names = {
            "enhanced_vnd_knn",   # [P0] K-NN based VND — primary improvement engine
            "hgs_fast_local_search",
            "segment_exchange_tail_search",
            "giant_tour_dp_split",
            "cross_route_2opt"
        }
        
        heavy_improvement_names = {
            "swap_star",
            "or_opt_segment_relocate"
        }
        
        breakout_map = {
            "mass_ruin": ["advanced_sisr_ruin", "sisr_ruin", "sisr_ruin_4a5b", "radial_ruin_3c4d", "random_ruin_1a2b"],
            "recreate": ["regret_insertion_2f3a", "min_cost_insertion_048f", "min_cost_insertion_3b2b"],
            "crossover": ["route_based_crossover_9f8a", "hgs_giant_tour_crossover"]
        }

        # 1. Classify standard pool
        from src.util.util import load_function
        for h_name in self.heuristic_pool_names:
            base_name = os.path.basename(h_name).replace(".py", "")
            func = load_function(h_name, problem=self.problem)
            
            if base_name in constructive_names:
                self.constructive_heuristics.append(func)
            elif base_name in fast_improvement_names:
                self.fast_improvement_heuristics.append(func)
                self.improvement_heuristics.append(func)
            elif base_name in heavy_improvement_names:
                self.heavy_improvement_heuristics.append(func)
                self.improvement_heuristics.append(func)
            
            # Map to breakout dictionary
            for key, variations in breakout_map.items():
                if base_name in variations:
                    if key not in self.breakout_heuristics:
                        self.breakout_heuristics[key] = []
                    self.breakout_heuristics[key].append(func)

        # Optional: only load pyvrp_blackbox for near-BK micro polish.
        if self.enable_micro_pyvrp_polish:
            try:
                pyvrp_func = load_function("src/problems/cvrp/heuristics/evolved_heuristics.part3/pyvrp_blackbox.py", problem=self.problem)
                self.breakout_heuristics["pyvrp_blackbox"] = pyvrp_func
            except Exception as e:
                self.logger(f"Could not load pyvrp_blackbox: {e}")

        # Force load direct_replace_solution to eliminate direct ReplaceSolutionOperator usage
        try:
            direct_replace_func = load_function("src/problems/cvrp/heuristics/evolved_heuristics.part3/direct_replace_solution.py", problem=self.problem)
            self.breakout_heuristics["direct_replace_solution"] = direct_replace_func
        except Exception as e:
            self.logger(f"Could not load direct_replace_solution: {e}")
        
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
        if not self.fast_improvement_heuristics: return False
        
        max_vnd_loops = 500  # Hard cap per VND call
        total_improved = False
        start_time = time.time()
        
        current_cost = self._get_pure_distance_cost(env) if self._is_feasible(env) else env.key_value
        _pool_best_vnd = min((s["value"] for s in self.elite_pool), default=float('inf')) if self.elite_pool else float('inf')
        is_near_bk = self._is_feasible(env) and current_cost <= _pool_best_vnd + 150.0

        heuristics_queue = list(self.fast_improvement_heuristics)
        if is_near_bk:
            heuristics_queue.extend(self.heavy_improvement_heuristics)
        
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

    def _get_retry_budget(self, env: Env, current_best: float, best_is_feasible: bool) -> int:
        """
        Near the best known solution we should avoid over-escalating into rebuilds.
        Give high-quality trajectories more chances to intensify before macro restarts.
        """
        node_num = env.instance_data.get("node_num", 80)
        base_budget = max(6, int(node_num * 0.06))

        if not best_is_feasible:
            return base_budget
        pool_best_rb = min((s["value"] for s in self.elite_pool), default=float('inf')) if self.elite_pool else float('inf')
        if pool_best_rb == float('inf'):
            return base_budget
        gap_to_pool = max(0.0, current_best - pool_best_rb)
        if gap_to_pool <= 30.0:
            return max(base_budget, 14)
        if gap_to_pool <= 100.0:
            return max(base_budget, 10)
        return base_budget
        

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
        """Strict feasibility: complete visit + no duplicates + capacity."""
        demands = env.instance_data['demands']
        capacity = env.instance_data['capacity']
        depot = env.instance_data.get('depot', 0)
        node_num = env.instance_data.get('node_num', 0)

        seen = set()
        for route in env.current_solution.routes:
            route_load = 0.0
            for n in route:
                if n == depot:
                    continue
                if n <= 0 or n >= node_num:
                    return False
                if n in seen:
                    return False
                seen.add(n)
                route_load += demands[n]
            if route_load > capacity + 1e-6:
                return False

        expected = node_num - 1
        return len(seen) == expected

    def _record_dual_population(self, env: Env):
        """Record current solution into feasible/infeasible sub-populations."""
        if not hasattr(env, 'current_solution') or env.current_solution is None:
            return

        entry = {
            'routes': [list(r) for r in env.current_solution.routes],
            'fingerprint': self._get_cvrp_fingerprint(env),
            'timestamp': time.time(),
        }

        if self._is_feasible(env):
            entry['value'] = self._get_pure_distance_cost(env)
            pool = self.feasible_subpop
            max_size = self.FEASIBLE_SUBPOP_MAX
        else:
            entry['value'] = env.key_value
            pool = self.infeasible_subpop
            max_size = self.INFEASIBLE_SUBPOP_MAX

        for item in pool:
            if abs(item['value'] - entry['value']) < 1e-6 and self._get_cvrp_distance(item['fingerprint'], entry['fingerprint']) < 5:
                return

        pool.append(entry)
        pool.sort(key=lambda x: x['value'])
        if len(pool) > max_size:
            del pool[max_size:]

    def _pick_crossover_target(self):
        """PyVRP-inspired parent choice using dual population."""
        # Prefer feasible parents, but keep a chance to pull an infeasible one
        # to inject boundary structure (quality-diversity tradeoff).
        if self.feasible_subpop and (not self.infeasible_subpop or random.random() < 0.78):
            top = self.feasible_subpop[:min(10, len(self.feasible_subpop))]
            return random.choice(top)

        if self.infeasible_subpop:
            top = self.infeasible_subpop[:min(6, len(self.infeasible_subpop))]
            return random.choice(top)

        if self.elite_pool:
            return random.choice(self.elite_pool[:min(8, len(self.elite_pool))])

        return None

    def _is_better_solution(self, env: Env, best_is_feasible: bool, best_value: float) -> tuple[bool, bool, float, float | None]:
        """
        Feasibility-first comparison:
        - Any feasible solution is better than infeasible best.
        - Among feasible solutions, compare pure distance cost.
        - Among infeasible solutions (only when no feasible best yet), compare penalized objective.
        """
        current_feasible = self._is_feasible(env)
        current_pure = self._get_pure_distance_cost(env) if current_feasible else None
        current_metric = current_pure if current_feasible else env.key_value

        if current_feasible and not best_is_feasible:
            return True, current_feasible, current_metric, current_pure

        if current_feasible and best_is_feasible:
            return (current_metric < best_value - 1e-3), current_feasible, current_metric, current_pure

        if (not current_feasible) and (not best_is_feasible):
            return (current_metric < best_value - 1e-3), current_feasible, current_metric, current_pure

        return False, current_feasible, current_metric, current_pure

    def _add_to_local_pool(self, env: Env, current_best: float):
        """Add solution to elite pool with strict diversity check. Only stores feasible solutions with pure distance cost."""
        # HGS: Only add feasible solutions to elite pool
        if not self._is_feasible(env):
            return False
            
        pure_cost = self._get_pure_distance_cost(env)

        # Quality gate: avoid flooding pool with weak feasible solutions after rebuild.
        # Keep only near-competitive solutions so L2 injection stays high-quality.
        if self.elite_pool:
            pool_best = self.elite_pool[0]['value']
            dynamic_margin = max(220.0, pool_best * 0.012)
            if pure_cost > pool_best + dynamic_margin:
                return False
        if hasattr(self, 'global_best_cost') and self.global_best_cost is not None:
            if pure_cost > self.global_best_cost + 350.0:
                return False

        fingerprint = self._get_cvrp_fingerprint(env)
        
                # Check against existing to maintain strict diversity (radius = 8 edges)
        is_duplicate = False
        for elite in self.elite_pool:
            if abs(elite['value'] - pure_cost) < 1e-4:
                dist = self._get_cvrp_distance(elite['fingerprint'], fingerprint)
                if dist < 8:
                    is_duplicate = True
                    break
            else:
                dist = self._get_cvrp_distance(elite['fingerprint'], fingerprint)
                # If they are practically identical (distance < 10) but different cost
                # Only keep the one with better cost to avoid clone swarms
                if dist < 10:
                    is_duplicate = True
                    if pure_cost < elite['value']:
                        elite['value'] = pure_cost
                        elite['fingerprint'] = fingerprint
                        if hasattr(env, 'current_solution'):
                            elite['routes'] = [list(r) for r in env.current_solution.routes]
                        elite['timestamp'] = time.time()
                        self.elite_pool.sort(key=lambda x: x['value'])
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

    def _apply_breakout(self, env: Env, strategy: str, current_best: float | None = None):
        # [DYNAMIC PENALTY LADDER START] Controlled, short-lived penalty drop.
        if strategy in ["elite_route_injection", "macro_route_ruin", "targeted_ruin"]:
            base_pf = float(getattr(env, "penalty_factor", 200.0))
            temp_pf = max(1.0, base_pf * 0.45)  # Floor lowered to match min_penalty
            env.problem_state["temporary_penalty_factor"] = temp_pf
            env.problem_state["temporary_penalty_steps"] = 8
            env.problem_state["capacity_penalty_factor"] = temp_pf
            env.penalty_factor = temp_pf
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
            if not env.validation_solution():
                self.logger("Targeted ruin produced invalid structure. Rolling back.")
                env.import_solution_wrapper(backup_wrapper)
                return
                
        elif strategy == "elite_route_injection":
            # [L2 - Elite Route Injection / Crossover]
            crossover_parent = self._pick_crossover_target()
            if len(self.elite_pool) < 2 and crossover_parent is None:
                self.logger("Elite pool too small for Crossover. Falling back to L1 targeted_ruin.")
                return self._apply_breakout(env, "targeted_ruin")
            
            _cur_cost_inj = self._get_pure_distance_cost(env) if self._is_feasible(env) else float('inf')
            _pool_best_inj = min((s["value"] for s in self.elite_pool), default=float('inf')) if self.elite_pool else float('inf')
            current_is_near_bk = self._is_feasible(env) and _cur_cost_inj <= _pool_best_inj + 80.0

            # Strategy mix: 50% crossover, 50% direct elite import + perturbation
            use_direct_import = (random.random() < (0.8 if current_is_near_bk else 0.5))
            
            # Find best elite that is different from current solution
            fingerprint = self._get_cvrp_fingerprint(env)
            candidate_pool = self.elite_pool[:self.ELITE_FILTER_SIZE] if hasattr(self, 'ELITE_FILTER_SIZE') else self.elite_pool
            
            candidates = []
            for elite in candidate_pool:
                dist = self._get_cvrp_distance(fingerprint, elite['fingerprint'])
                if dist >= 3:  # Lower threshold to accept more candidates
                    candidates.append((dist, elite))
            
            if not candidates and crossover_parent is None:
                self.logger("Active Relinking: All elites too similar. Falling back to L1.")
                return self._apply_breakout(env, "targeted_ruin")

            # Guard: direct-import needs at least one candidate; otherwise
            # fallback to crossover branch to avoid empty-choice crashes.
            if use_direct_import and not candidates:
                use_direct_import = False
            
            if use_direct_import:
                # === DIRECT ELITE IMPORT === 
                # Pick the BEST quality elite (not most distant) and start VND from there
                # This is the key HGS mechanism: educate offspring from best parents
                best_candidates = sorted(candidates, key=lambda x: x[1]['value'])[:(3 if current_is_near_bk else 5)]
                _, target_elite = random.choice(best_candidates)
                
                backup_wrapper = env.export_solution_wrapper()
                
                target_routes = [list(r) for r in target_elite['routes']]
                direct_h = self.breakout_heuristics.get("direct_replace_solution")
                if direct_h:
                    env.run_heuristic(direct_h, parameters={"target_routes": target_routes})
                else:
                    self.logger("Error: direct_replace_solution missing, rollback.")
                    env.import_solution_wrapper(backup_wrapper)
                    return
                
                # Apply small perturbation (5-10% ruin) so VND can find new improving moves
                ratio = random.uniform(0.03, 0.06) if current_is_near_bk else random.uniform(0.05, 0.10)
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
                    if not env.validation_solution():
                        self.logger("Direct-import branch produced invalid structure. Rolling back.")
                        env.import_solution_wrapper(backup_wrapper)
                        return
            else:
                # === CROSSOVER INJECTION (original L2) ===
                if crossover_parent is not None:
                    target_elite_dict = crossover_parent
                else:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    top_candidates = candidates[:min(3, len(candidates))]
                    chosen_dist, target_elite_dict = random.choice(top_candidates)
                
                # Pass target routes cleanly without direct Solution class dependencies
                class TargetSolutionStub:
                    def __init__(self, routes):
                        self.routes = routes
                target_sol = TargetSolutionStub(target_elite_dict['routes'])
                
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
                if not env.validation_solution():
                    self.logger("Crossover branch produced invalid structure. Rolling back.")
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
            if not env.validation_solution():
                self.logger("Macro ruin produced invalid structure. Rolling back.")
                env.import_solution_wrapper(backup_wrapper)
                return

        elif strategy == "bk_plateau_shake":
            # PyVRP-inspired plateau escape:
            # apply a stronger controlled ruin on a high-quality incumbent,
            # then recreate and immediately educate.
            ratio = random.uniform(0.16, 0.28)
            ruin_h = random.choice(self.breakout_heuristics["mass_ruin"]) if "mass_ruin" in self.breakout_heuristics else None
            recreate_h = random.choice(self.breakout_heuristics["recreate"]) if "recreate" in self.breakout_heuristics else None
            if not ruin_h or not recreate_h:
                self.logger("Warning: Missing operators for bk_plateau_shake. Falling back to targeted_ruin.")
                return self._apply_breakout(env, "targeted_ruin")

            backup_wrapper = env.export_solution_wrapper()
            try:
                env.run_heuristic(ruin_h, parameters={"removal_fraction": ratio})
            except Exception:
                env.import_solution_wrapper(backup_wrapper)
                return

            c_steps = 0
            while not env.is_complete_solution and c_steps < 120:
                try:
                    op = env.run_heuristic(recreate_h)
                    if not op or isinstance(op, str):
                        break
                except Exception:
                    break
                c_steps += 1

            if not env.is_complete_solution or not env.validation_solution():
                env.import_solution_wrapper(backup_wrapper)
                return

            # Immediate education to exploit the new basin.
            self._run_improvement_phase(env, time_limit=8.0)

        elif strategy == "pyvrp_reseed":
            # Partial usage only: near-BK tiny polish; fallback to internal shake.
            polished = self._try_micro_pyvrp_polish(env, current_best=current_best)
            if polished:
                self._run_improvement_phase(env, time_limit=6.0)
                return
            return self._apply_breakout(env, "bk_plateau_shake")
                
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
                       target_routes = [list(r) for r in target_elite['routes']]
                       direct_h = self.breakout_heuristics.get("direct_replace_solution")
                       if direct_h:
                           env.run_heuristic(direct_h, parameters={"target_routes": target_routes})
                           self.logger(f"Restarted from Distant Elite (Val: {target_elite['value']}, Dist: {target_dist})")
                       else:
                           self.logger(f"Error: direct_replace_solution missing. Forcing constructive.")
                           force_constructive = True
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
        target_feasible_ratio = 0.5   # HGS default: aim for ~50% feasible (Vidal 2012)
        adapt_factor = 1.2  # Moderate adjustment (HGS uses 1.2)
        min_penalty = 5.0   # Floor: prevents route-merging while allowing moderate infeasibility
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
            return False
            
        current_ratio = sum(self._feasibility_history) / len(self._feasibility_history)
        current_pf = getattr(env, 'penalty_factor', 200.0)
        
        if current_ratio > target_feasible_ratio + 0.05:
            # Too many feasible → decrease penalty to explore infeasible space
            new_pf = max(min_penalty, current_pf / adapt_factor)
        elif current_ratio < target_feasible_ratio - 0.05:
            # Too few feasible → increase penalty to push back
            new_pf = min(max_penalty, current_pf * adapt_factor)
        else:
            return False  # Within target range, no adjustment needed
        
        env.penalty_factor = new_pf
        env.problem_state["capacity_penalty_factor"] = new_pf
        return True


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
        
        # [MIXED INIT STRATEGY] Decide whether to use PyVRP or pure construction for this epoch
        # 只允许pure construction初始化
        use_pyvrp_this_epoch = False
        seeded = False
        self.logger(f"[INIT] Worker {self.worker_id} uses PURE CONSTRUCTION (no PyVRP)")
        
        if not seeded:
            # Fallback loop until solution is COMPLETE and VALID
            max_retries = 10
            for retry in range(max_retries):
                # --- Phase A: Cold Start (Initialization) ---
                env.clear_solution()

                construction_steps = 0
                prev_unvisited = 1000
                stagnation_counter = 0

                # Pick ONE random constructive heuristic to build the entire solution in this attempt (prevents chaotic mixed routes)
                if not self.constructive_heuristics:
                    self.logger("Critical Failure: No constructive heuristics found.")
                    return False
                current_h = random.choice(self.constructive_heuristics)

                # CVRP construct loop until solution is complete (all nodes visited and legally routed)
                while not env.is_complete_solution and construction_steps < 1000:
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
                        # Keep using the same constructive heuristic for the whole phase
                        try:
                            env.run_heuristic(current_h)
                        except Exception:
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
        else:
            # Small education pass right after PyVRP seeding.
            self._run_improvement_phase(env, time_limit=8.0)

        best_is_feasible = self._is_feasible(env)
        current_best = self._get_pure_distance_cost(env) if best_is_feasible else env.key_value
        best_wrapper = env.export_solution_wrapper()
        no_improve_steps = 0
        self.current_run_steps = 0
        self.stagnation_level = 0
        self.phase_retries = 0
        self.last_restart_step = 0
        
        while env.continue_run:
            self.current_run_steps += 1
            
            # --- HGS: Adapt penalty factor before improvement ---
            penalty_changed = self._adapt_penalty_factor(env)
            if penalty_changed and best_wrapper is not None:
                # Evaluate the best_wrapper under the new penalty landscape
                curr_sol_wrapper = env.export_solution_wrapper()
                env.import_solution_wrapper(best_wrapper)
                current_best = env.key_value
                env.import_solution_wrapper(curr_sol_wrapper)

            # Maintain dual population continuously (PyVRP-style).
            self._record_dual_population(env)
            
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
            
            is_better, is_sol_feas, candidate_metric, pure_cost = self._is_better_solution(env, best_is_feasible, current_best)
            if is_better:
                old_best = current_best
                current_best = candidate_metric
                best_is_feasible = is_sol_feas
                best_wrapper = env.export_solution_wrapper()
                
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

                # Polish is triggered only via stagnation ladder (L4), not here.
                
                # Log state (aligned format)
                feas_tag = "" if is_sol_feas else " [INFEASIBLE]"
                cost_display = f"{pure_cost:.0f}" if is_sol_feas else f"{candidate_metric:.0f}*"
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
            patience = 0
            
            if no_improve_steps > patience:
                max_retries_per_phase = self._get_retry_budget(env, current_best, best_is_feasible)
                self.phase_retries += 1
                
                # Check relation to Elite Pool (Global Best)
                is_attacking_global_best = False
                global_best_val = float('inf')
                if self.elite_pool:
                     global_best_val = min(s["value"] for s in self.elite_pool)
                     # Using tolerance for CVRP (minimization)
                     if current_best <= global_best_val + 1e-3:
                         is_attacking_global_best = True

                close_to_pool_best = best_is_feasible and current_best <= global_best_val + 140.0

                if close_to_pool_best and len(self.elite_pool) >= 2:
                    if self.stagnation_level < 2:
                        self.stagnation_level = 2
                        self.phase_retries = 1

                    # Use stagnation_level to track sub-level, so phase_retries reset doesn't cycle back.
                    if self.stagnation_level == 2 and self.phase_retries <= max_retries_per_phase:
                        strategy = "elite_route_injection"
                    elif self.stagnation_level == 2 and self.phase_retries > max_retries_per_phase:
                        strategy = "bk_plateau_shake"
                        self.stagnation_level = 3
                        self.phase_retries = 1
                        self.logger(
                            f"Escalating to bk_plateau_shake near pool best (best={current_best:.0f}, pool={global_best_val:.0f})"
                        )
                    elif self.stagnation_level == 3 and self.phase_retries <= max_retries_per_phase:
                        strategy = "bk_plateau_shake"
                    elif self.stagnation_level == 3 and self.phase_retries > max_retries_per_phase:
                        strategy = "pyvrp_reseed"
                        self.stagnation_level = 4
                        self.phase_retries = 1
                        self.logger(
                            f"Escalating to pyvrp_reseed near pool best (best={current_best:.0f}, pool={global_best_val:.0f})"
                        )
                    elif self.stagnation_level == 4 and self.phase_retries <= max_retries_per_phase:
                        strategy = "pyvrp_reseed"
                    elif self.stagnation_level == 4 and self.phase_retries > max_retries_per_phase:
                        self.stagnation_level = 5
                        self.phase_retries = 1
                        self.logger(
                            f"Escalating to global_rebuild near pool best (best={current_best:.0f}, pool={global_best_val:.0f})"
                        )
                        strategy = "pyvrp_reseed"  # overridden by stagnation_level>=5 block below
                    else:
                        strategy = "pyvrp_reseed"
                elif self.stagnation_level == 0:
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
                    strategy = "soft_restart" if close_to_pool_best else "macro_route_ruin"
                    self.logger(f"Escalating Stagnation Level to {self.stagnation_level} (Exhausted L2 retries)")
                elif self.stagnation_level == 3 and self.phase_retries <= max_retries_per_phase:
                    strategy = "soft_restart" if close_to_pool_best else "macro_route_ruin"
                elif self.stagnation_level == 3 and self.phase_retries > max_retries_per_phase:
                    # Upgrade to L4: micro polish (falls back to bk_plateau_shake when budget exhausted)
                    self.stagnation_level = 4
                    self.phase_retries = 1
                    strategy = "pyvrp_reseed"
                    self.logger(f"Escalating Stagnation Level to {self.stagnation_level} (Exhausted L3 retries, trying micro polish)")
                elif self.stagnation_level == 4 and self.phase_retries <= max_retries_per_phase:
                    strategy = "pyvrp_reseed"
                else:
                    self.stagnation_level += 1
                    self.phase_retries = 1
                    self.logger(f"Escalating Stagnation Level to {self.stagnation_level} (Exhausted L4 retries)")
                    strategy = "soft_restart"
                    
                # Evaluate Hard Restarts explicitly
                if self.stagnation_level >= 5:
                    if is_attacking_global_best or close_to_pool_best:
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
                self._apply_breakout(env, strategy, current_best=current_best)
                
                # If L4 triggers, worker abandons trajectory. We MUST reset current_best to track the new trajectory.
                if strategy == "soft_restart":
                    if is_attacking_global_best:
                        self.logger("Leader Mode: Retaining high 'current_best' baseline to force meaningful improvement across Soft Restarts.")
                    else:
                        self.logger("Follower Mode: Resetting 'current_best' to allow local hill climbing.")
                        best_is_feasible = self._is_feasible(env)
                        current_best = self._get_pure_distance_cost(env) if best_is_feasible else env.key_value
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

                rebuild_seed_entries = []
                if self.elite_pool:
                    for elite in self.elite_pool[:self.REBUILD_SEED_COUNT]:
                        rebuild_seed_entries.append({
                            'value': elite['value'],
                            'fingerprint': elite['fingerprint'],
                            'routes': [list(route) for route in elite['routes']],
                            'timestamp': elite['timestamp'],
                        })

                if hasattr(self, 'global_best_wrapper') and getattr(self, 'global_best_wrapper') is not None:
                    env.import_solution_wrapper(self.global_best_wrapper)
                    if self._is_feasible(env):
                        global_seed = {
                            'value': self._get_pure_distance_cost(env),
                            'fingerprint': self._get_cvrp_fingerprint(env),
                            'routes': [list(route) for route in env.current_solution.routes],
                            'timestamp': time.time(),
                        }
                        if not rebuild_seed_entries:
                            rebuild_seed_entries.append(global_seed)
                        elif self._get_cvrp_distance(rebuild_seed_entries[0]['fingerprint'], global_seed['fingerprint']) >= 4:
                            rebuild_seed_entries.insert(0, global_seed)
                
                # [L5 CORE: Reset Environment Logic for CVRP - Scorched Earth]
                env.reset()
                
                # pool_id already incremented in Phase D breakout block using Atomic Filesys Create
                
                self.elite_pool = [] # Destroy all locally accumulated Elite solutions

                for entry in rebuild_seed_entries[:self.REBUILD_SEED_COUNT]:
                    try:
                        direct_h = self.breakout_heuristics.get("direct_replace_solution")
                        if direct_h:
                            env.run_heuristic(direct_h, parameters={"target_routes": [list(route) for route in entry['routes']]})
                            self._add_to_local_pool(env, entry['value'])
                        else:
                            self.logger("Error: direct_replace_solution missing during rebuild.")
                    except Exception:
                        continue

                env.reset()
                    
                self.pending_rebuild = False
                continue
                
            return result
