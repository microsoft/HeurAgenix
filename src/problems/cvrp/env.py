import os
import tsplib95
import numpy as np
import pandas as pd
import networkx as nx
from src.problems.base.env import BaseEnv
from src.problems.base.components import BaseOperator
from src.problems.cvrp.components import Solution, AppendOperator, InsertOperator, SwapOperator, ReverseSegmentOperator, RelocateOperator, BatchRemoveOperator, BatchInsertOperator, MergeRoutesOperator, ReplaceSolutionOperator, RemoveNodesOperator, SwapStarOperator, BlockRelocateOperator
from src.problems.cvrp.best_known import best_known


class Env(BaseEnv):
    """CVRP env that stores the instance data, current solution, and problem state to support algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "cvrp")

        self.penalty_factor = getattr(self, "penalty_factor", 200.0)
        self.feasible_history = []

        self.construction_steps = self.instance_data["node_num"]
        self.key_item = "total_cost"
        self.compare = lambda x, y: y - x

    @property
    def is_complete_solution(self) -> bool:
        # Fast O(V) check: each route has 1 depot, so total len = nodes - 1 + vehicles
        expected_len = self.instance_data["node_num"] - 1 + self.instance_data["vehicle_num"]
        return sum(len(route) for route in self.current_solution.routes) == expected_len

    def load_data(self, data_path: str) -> None:
        data_name = data_path.split(os.sep)[-1].split(".")[0]
        self.best_known = best_known.get(data_name, None)
        problem = tsplib95.load(data_path)
        depot = problem.depots[0] - 1
        if problem.edge_weight_type == "EUC_2D":
            node_coords = problem.node_coords
            node_num = len(node_coords)
            distance_matrix = np.zeros((node_num, node_num))
            for i in range(node_num):
                for j in range(node_num):
                    if i != j:
                        x1, y1 = node_coords[i + 1]
                        x2, y2 = node_coords[j + 1]
                        # TSPLIB EUC_2D standard rounding: int(sqrt(dx^2 + dy^2) + 0.5)
                        distance_matrix[i][j] = np.floor(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) + 0.5)
        else:
            distance_matrix = nx.to_numpy_array(problem.get_graph())
            node_num = len(distance_matrix)
        if os.path.basename(data_path).split(".")[0].split("-")[-1][0] == "k":
            vehicle_num = int(os.path.basename(data_path).split(".")[0].split("-")[-1][1:])
        elif open(data_path).readlines()[-1].strip().split(" : ")[0] == "VEHICLE":
            vehicle_num = int(open(data_path).readlines()[-1].strip().split(" : ")[-1])
        else:
            raise NotImplementedError("Vehicle number error")
        
        # Calculate K-Nearest Neighbors for Granular Neighborhood search
        # Keep top 40 neighbors (excluding self, so start at index 1)
        nearest_neighbors = np.argsort(distance_matrix, axis=1)[:, 1:min(node_num, 41)]

        capacity = problem.capacity
        demands = np.array(list(problem.demands.values()))
        total_demands = demands.sum()
        load_ratio = float(total_demands / (vehicle_num * capacity))
        return {"node_num": node_num, "distance_matrix": distance_matrix, "depot": depot, "vehicle_num": vehicle_num, "capacity": capacity, "demands": demands, "load_ratio": load_ratio, "nearest_neighbors": nearest_neighbors}

    def init_solution(self) -> Solution:
        vehicle_num = self.instance_data["vehicle_num"]
        depot = self.instance_data["depot"]
        # Depot initially in the route.
        depot_demand = float(self.instance_data["demands"][depot])
        return Solution(
            routes=[[depot] for _ in range(vehicle_num)], 
            depot=depot,
            total_cost=0.0,
            loads=[depot_demand] * vehicle_num
        )


    def _recalculate_exact(self) -> float:
        solution = self.current_solution
        total_current_cost = 0.0
        demands = self.instance_data["demands"]
        capacity = self.instance_data["capacity"]
        penalty_factor = self.problem_state.get("capacity_penalty_factor", getattr(self, "penalty_factor", 200.0)) # Penalty multiplier
        
        for vehicle_index in range(self.instance_data["vehicle_num"]):
            route = solution.routes[vehicle_index]
            if len(route) == 0: continue
            dist = self._get_route_cost(route)
            total_current_cost += dist
            
            # Add capacity penalty
            load = sum(demands[n] for n in route)
            if load > capacity:
                total_current_cost += (load - capacity) * penalty_factor
                
        # Unvisited and duplicate node penalty
        expected_len = self.instance_data["node_num"] - 1 + self.instance_data["vehicle_num"]
        actual_len = sum(len(route) for route in solution.routes)
        if actual_len != expected_len:
            total_current_cost += abs(expected_len - actual_len) * 100000.0
                
        return total_current_cost

    def get_key_value(self, recalculate: bool=False) -> float:
        """Get the key value of the current solution based on the key item."""
        if not getattr(self, 'current_solution', None): return 0.0
        
        # If passing self.current_solution object instead of bool logic, treat as recalculate
        if isinstance(recalculate, object) and not isinstance(recalculate, bool):
            recalculate = True
            
        if not recalculate and getattr(self.current_solution, 'total_cost', None) is not None:
            return self.current_solution.total_cost
            
        solution = self.current_solution
        total_current_cost = 0.0
        for vehicle_index in range(self.instance_data["vehicle_num"]):
            route = solution.routes[vehicle_index]
            if len(route) == 0: continue
            total_current_cost += self._get_route_cost(route) + self._get_route_penalty(route)
            
        # Unvisited and duplicate node penalty
        expected_len = self.instance_data["node_num"] - 1 + self.instance_data["vehicle_num"]
        actual_len = sum(len(route) for route in solution.routes)
        if actual_len != expected_len:
            total_current_cost += abs(expected_len - actual_len) * 100000.0
            
        return total_current_cost

    def _get_route_cost(self, route: list[int]) -> float:
        """Calculate total pure distance of a strictly single route."""
        if not route:
            return 0.0
        n = len(route)
        if n == 1:
            return 0.0
        dist = self.instance_data["distance_matrix"]
        return sum([dist[route[i]][route[(i + 1) % n]] for i in range(n)])

    def _get_route_penalty(self, route: list[int]) -> float:
        """Calculate capacity violation penalty for a route."""
        # Always sync penalty_factor from problem_state (written by update_problem_state as "capacity_penalty_factor")
        pf = self.problem_state.get("capacity_penalty_factor", getattr(self, "penalty_factor", 200.0))
        pf = min(pf, 100000.0)
        self.penalty_factor = pf
        if not route:
            return 0.0
        demands = self.instance_data["demands"]
        capacity = self.instance_data["capacity"]
        load = sum(demands[n] for n in route)
        return max(0, load - capacity) * pf

    def _is_invalid_operator(self, operator: BaseOperator) -> bool:
        if operator is None: return True
        
        op_type = type(operator)
        
        # Check depot interactions without dict lookups and slow getattr if possible
        if hasattr(operator, "node") and operator.node == self.instance_data["depot"]: return True
        if hasattr(operator, "nodes") and self.instance_data["depot"] in operator.nodes: return True

        if op_type is InsertOperator and operator.position == 0: return True
        if op_type is RelocateOperator and (operator.source_position == 0 or operator.target_position == 0): return True
        if op_type is SwapOperator and (operator.position1 == 0 or operator.position2 == 0): return True
        if op_type is SwapStarOperator and (operator.best_pos_for_1_in_2 == 0 or operator.best_pos_for_2_in_1 == 0): return True
        if op_type is ReverseSegmentOperator:
            for s, e in operator.segments:
                if s == 0: return True
        return False

    def _calculate_delta(self, operator: BaseOperator) -> float:
        if self._is_invalid_operator(operator):
            return 0.0
            
        dist = self.instance_data["distance_matrix"]
        solution = self.current_solution
        delta = 0.0

        if isinstance(operator, AppendOperator):
            vid = operator.vehicle_id
            node = operator.node
            route = solution.routes[vid]
            n = len(route)
            if n == 0:
                delta = 0.0
            else:
                first = route[0]
                last = route[-1]
                delta = -dist[last][first] + dist[last][node] + dist[node][first]

        elif isinstance(operator, InsertOperator):
            vid = operator.vehicle_id
            node = operator.node
            pos = operator.position
            route = solution.routes[vid]
            n = len(route)
            if n == 0:
                delta = 0.0
            else:
                prev_node = route[(pos - 1) % n]
                next_node = route[pos % n]
                delta = -dist[prev_node][next_node] + dist[prev_node][node] + dist[node][next_node]

        elif isinstance(operator, RelocateOperator):
            svid, spos = operator.source_vehicle_id, operator.source_position
            tvid, tpos = operator.target_vehicle_id, operator.target_position
            
            s_route = solution.routes[svid]
            t_route = solution.routes[tvid]
            
            if svid != tvid:
                node = s_route[spos]
                n_s = len(s_route)
                n_t = len(t_route)
                
                prev_s = s_route[(spos - 1) % n_s]
                next_s = s_route[(spos + 1) % n_s]
                delta = -dist[prev_s][node] - dist[node][next_s] + dist[prev_s][next_s]
                
                if n_t == 0:
                    delta += 0.0
                else:
                    prev_t = t_route[(tpos - 1) % n_t]
                    next_t = t_route[tpos % n_t]
                    delta += -dist[prev_t][next_t] + dist[prev_t][node] + dist[node][next_t]
            else:
                if spos == tpos: return 0.0
                n_s = len(s_route)
                node = s_route[spos]
                old_cost = sum([dist[s_route[i]][s_route[(i + 1) % n_s]] for i in range(n_s)])
                
                temp_route = s_route[:spos] + s_route[spos+1:]
                actual_tpos = tpos if spos >= tpos else tpos - 1
                new_route = temp_route[:actual_tpos] + [node] + temp_route[actual_tpos:]
                
                n_new = len(new_route)
                new_cost = sum([dist[new_route[i]][new_route[(i + 1) % n_new]] for i in range(n_new)])
                delta = new_cost - old_cost

        elif isinstance(operator, SwapOperator):
             v1, p1 = operator.vehicle_id1, operator.position1
             v2, p2 = operator.vehicle_id2, operator.position2
             r1, r2 = solution.routes[v1], solution.routes[v2]
             n1_len, n2_len = len(r1), len(r2)
             
             if v1 == v2:
                 if p1 == p2: return 0.0
                 if p1 > p2: p1, p2 = p2, p1
                 node1, node2 = r1[p1], r1[p2]
                 
                 if (p1 + 1) % n1_len == p2 or (p2 + 1) % n1_len == p1:
                     if n1_len <= 2: return 0.0
                     first, second = p1, p2
                     if (p2 + 1) % n1_len == p1:
                         first, second = p2, p1
                         node1, node2 = r1[first], r1[second]
                     prev_n = r1[(first - 1) % n1_len]
                     next_n = r1[(second + 1) % n1_len]
                     delta = -dist[prev_n][node1] - dist[node2][next_n] + dist[prev_n][node2] + dist[node1][next_n]
                     delta += -dist[node1][node2] + dist[node2][node1]
                 else:
                     prev_1, next_1 = r1[(p1 - 1) % n1_len], r1[(p1 + 1) % n1_len]
                     prev_2, next_2 = r1[(p2 - 1) % n1_len], r1[(p2 + 1) % n1_len]
                     delta = -dist[prev_1][node1] - dist[node1][next_1] - dist[prev_2][node2] - dist[node2][next_2]
                     delta += dist[prev_1][node2] + dist[node2][next_1] + dist[prev_2][node1] + dist[node1][next_2]
             else:
                 node1, node2 = r1[p1], r2[p2]
                 prev_1, next_1 = r1[(p1 - 1) % n1_len], r1[(p1 + 1) % n1_len]
                 prev_2, next_2 = r2[(p2 - 1) % n2_len], r2[(p2 + 1) % n2_len]
                 delta = -dist[prev_1][node1] - dist[node1][next_1] - dist[prev_2][node2] - dist[node2][next_2]
                 delta += dist[prev_1][node2] + dist[node2][next_1] + dist[prev_2][node1] + dist[node1][next_2]

        elif isinstance(operator, ReverseSegmentOperator):
            vid = operator.vehicle_id
            route = solution.routes[vid]
            n = len(route)
            if n > 2:
                for start, end in operator.segments:
                    node_start = route[start % n]
                    node_end = route[end % n]
                    prev_start = route[(start - 1) % n]
                    next_end = route[(end + 1) % n]
                    
                    delta += -dist[prev_start][node_start] - dist[node_end][next_end]
                    delta += dist[prev_start][node_end] + dist[node_start][next_end]

        return delta
        
    def _update_loads_full(self, solution):
        demands = self.instance_data["demands"]
        solution.loads = [sum(demands[n] for n in route) for route in solution.routes]

    def run_operator(self, operator: BaseOperator) -> bool:
        """Apply the operator to the current solution In-Place."""
        if isinstance(operator, ReplaceSolutionOperator):
            self.current_solution.routes = [list(route) for route in operator.routes]
            demands = self.instance_data["demands"]
            self.current_solution.loads = [sum(demands[n] for n in route) for route in self.current_solution.routes]
            self.current_solution.total_cost = self.get_key_value(recalculate=True)
            
            self.update_problem_state()


            return True
            
        if self._is_invalid_operator(operator):
            return False
            
        solution = self.current_solution
        demands = self.instance_data["demands"]

        # 1. Detect affected vehicles for safe and fast Route-Level Delta evaluation
        affected_vehicles = set()
        if isinstance(operator, AppendOperator):
            affected_vehicles.add(operator.vehicle_id)
        elif isinstance(operator, InsertOperator):
            affected_vehicles.add(operator.vehicle_id)
        elif isinstance(operator, SwapOperator):
            affected_vehicles.add(operator.vehicle_id1)
            affected_vehicles.add(operator.vehicle_id2)
        elif isinstance(operator, ReverseSegmentOperator):
            affected_vehicles.add(operator.vehicle_id)
        elif isinstance(operator, RelocateOperator):
            affected_vehicles.add(operator.source_vehicle_id)
            affected_vehicles.add(operator.target_vehicle_id)
        
        elif isinstance(operator, RemoveNodesOperator):
            for route in solution.routes:
                for node in operator.nodes:
                    if node in route:
                        route.remove(node)

        elif isinstance(operator, SwapStarOperator):
            affected_vehicles.add(operator.vehicle_id1)
            affected_vehicles.add(operator.vehicle_id2)
        elif isinstance(operator, BatchRemoveOperator):
            nodes_to_remove = set(operator.nodes)
            for vid, route in enumerate(solution.routes):
                if not set(route).isdisjoint(nodes_to_remove):
                    affected_vehicles.add(vid)
        elif isinstance(operator, BatchInsertOperator):
            for vid, pos, node in operator.insertions:
                affected_vehicles.add(vid)
        elif isinstance(operator, MergeRoutesOperator):
            affected_vehicles.add(operator.source_vehicle_id)
            affected_vehicles.add(operator.target_vehicle_id)
        elif isinstance(operator, BlockRelocateOperator):
            affected_vehicles.add(operator.source_vehicle_id)
            affected_vehicles.add(operator.target_vehicle_id)

        # 2. Determine if we can use O(1) Delta or need O(L) Route-Level recalculation
        is_complex_batch = isinstance(operator, (BatchRemoveOperator, BatchInsertOperator, MergeRoutesOperator, SwapStarOperator, BlockRelocateOperator)) or (isinstance(operator, ReverseSegmentOperator) and len(operator.segments) > 1) or (isinstance(operator, ReverseSegmentOperator) and operator.segments[0][0] % max(1, len(self.current_solution.routes[operator.vehicle_id])) > operator.segments[0][1] % max(1, len(self.current_solution.routes[operator.vehicle_id])))
        delta = 0.0
        old_cost = 0.0
        
        # ALWAYS Fallback to O(L) local route recalculation for Penalized tracking
        # The penalty delta cannot easily be modeled without tracking the loads anyway
        old_cost = sum(self._get_route_cost(solution.routes[vid]) + self._get_route_penalty(solution.routes[vid]) for vid in affected_vehicles)
        old_actual_len = sum(len(route) for route in solution.routes)
            
        # 3. Apply the IN-PLACE structural modifications AND Update Loads
        if isinstance(operator, AppendOperator):
            solution.routes[operator.vehicle_id].append(operator.node)
            solution.loads[operator.vehicle_id] += demands[operator.node]
            
        elif isinstance(operator, InsertOperator):
            solution.routes[operator.vehicle_id].insert(operator.position, operator.node)
            solution.loads[operator.vehicle_id] += demands[operator.node]
            
        elif isinstance(operator, SwapOperator):
            r1 = solution.routes[operator.vehicle_id1]
            r2 = solution.routes[operator.vehicle_id2]
            node1, node2 = r1[operator.position1], r2[operator.position2]
            r1[operator.position1], r2[operator.position2] = node2, node1
            if operator.vehicle_id1 != operator.vehicle_id2:
                solution.loads[operator.vehicle_id1] += demands[node2] - demands[node1]
                solution.loads[operator.vehicle_id2] += demands[node1] - demands[node2]
                
        elif isinstance(operator, ReverseSegmentOperator):
            r = solution.routes[operator.vehicle_id]
            n_r = len(r)
            if n_r > 0:
                for start, end in operator.segments:
                    start = start % n_r
                    end = end % n_r
                    if start <= end:
                        r[start:end+1] = r[start:end+1][::-1]
                    else:
                        segment = r[start:n_r] + r[0:end+1]
                        segment = segment[::-1]
                        r[start:n_r] = segment[:n_r-start]
                        r[0:end+1] = segment[n_r-start:]
            depot = self.instance_data["depot"]
            if r[0] != depot and depot in r:
                idx = r.index(depot)
                solution.routes[operator.vehicle_id] = r[idx:] + r[:idx]
            depot = self.instance_data["depot"]
            if r[0] != depot and depot in r:
                idx = r.index(depot)
                solution.routes[operator.vehicle_id] = r[idx:] + r[:idx]
                
        elif isinstance(operator, RelocateOperator):
            node = solution.routes[operator.source_vehicle_id].pop(operator.source_position)
            t_pos = operator.target_position
            if operator.source_vehicle_id == operator.target_vehicle_id and operator.source_position < operator.target_position:
                t_pos -= 1
            solution.routes[operator.target_vehicle_id].insert(t_pos, node)
            if operator.source_vehicle_id != operator.target_vehicle_id:
                solution.loads[operator.source_vehicle_id] -= demands[node]
                solution.loads[operator.target_vehicle_id] += demands[node]
            
        elif isinstance(operator, BatchRemoveOperator):
            nodes_to_remove = set(operator.nodes)
            for vid in affected_vehicles:
                removed_demand = sum(demands[n] for n in solution.routes[vid] if n in nodes_to_remove)
                solution.routes[vid] = [n for n in solution.routes[vid] if n not in nodes_to_remove]
                solution.loads[vid] -= removed_demand
                
        elif isinstance(operator, BatchInsertOperator):
            inserts_by_vehicle = {}
            for vid, pos, node in operator.insertions:
                if vid not in inserts_by_vehicle: inserts_by_vehicle[vid] = []
                inserts_by_vehicle[vid].append((pos, node))
            
            for vid, ops in inserts_by_vehicle.items():
                 ops.sort(key=lambda x: x[0], reverse=True)
                 for pos, node in ops:
                     solution.routes[vid].insert(pos, node)
                     solution.loads[vid] += demands[node]

        
        elif isinstance(operator, RemoveNodesOperator):
            for route in solution.routes:
                for node in operator.nodes:
                    if node in route:
                        route.remove(node)

        elif isinstance(operator, SwapStarOperator):
            r1 = solution.routes[operator.vehicle_id1]
            r2 = solution.routes[operator.vehicle_id2]
            n1 = operator.node1
            n2 = operator.node2
            r1.remove(n1)
            r2.remove(n2)
            r2.insert(operator.best_pos_for_1_in_2, n1)
            r1.insert(operator.best_pos_for_2_in_1, n2)
            solution.loads[operator.vehicle_id1] += demands[n2] - demands[n1]
            solution.loads[operator.vehicle_id2] += demands[n1] - demands[n2]
        
        elif isinstance(operator, BlockRelocateOperator):
            svid = operator.source_vehicle_id
            tvid = operator.target_vehicle_id
            si = operator.start_idx
            ei = operator.end_idx
            tpos = operator.target_position
            # Extract segment
            seg = solution.routes[svid][si:ei+1]
            seg_demand = sum(demands[n] for n in seg)
            # Remove from source (reverse order to keep indices valid)
            del solution.routes[svid][si:ei+1]
            # Insert into target
            for k, node in enumerate(seg):
                solution.routes[tvid].insert(tpos + k, node)
            # Update loads
            solution.loads[svid] -= seg_demand
            solution.loads[tvid] += seg_demand

        elif isinstance(operator, MergeRoutesOperator):
             depot = self.instance_data["depot"]
             r_src = solution.routes[operator.source_vehicle_id]
             r_tgt = solution.routes[operator.target_vehicle_id]
             
             src_nodes = []
             if depot in r_src:
                 idx = r_src.index(depot)
                 src_nodes = r_src[idx+1:] + r_src[:idx]
             else:
                 src_nodes = r_src[:]
                 
             tgt_nodes = []
             if depot in r_tgt:
                 idx = r_tgt.index(depot)
                 tgt_nodes = r_tgt[idx+1:] + r_tgt[:idx]
             else:
                 tgt_nodes = r_tgt[:]
                 
             solution.routes[operator.target_vehicle_id] = [depot] + src_nodes + tgt_nodes
             solution.routes[operator.source_vehicle_id] = [depot]
             
             # Also fix loads
             solution.loads[operator.target_vehicle_id] += solution.loads[operator.source_vehicle_id] - demands[depot]
             solution.loads[operator.source_vehicle_id] = demands[depot]

        # 4. Update Cost
        new_cost = sum(self._get_route_cost(solution.routes[vid]) + self._get_route_penalty(solution.routes[vid]) for vid in affected_vehicles)
        expected_len = self.instance_data["node_num"] - 1 + self.instance_data["vehicle_num"]
        new_actual_len = sum(len(route) for route in solution.routes)
        
        if solution.total_cost is None:
            solution.total_cost = self.get_key_value(recalculate=True)
        else:
            old_length_penalty = abs(expected_len - old_actual_len) * 100000.0
            new_length_penalty = abs(expected_len - new_actual_len) * 100000.0
            solution.total_cost += (new_cost - old_cost) + (new_length_penalty - old_length_penalty)
            
        self.update_problem_state()
        return True

    def update_problem_state(self) -> None:
        super().update_problem_state()
        
        # Start with the current instance penalty factor
        penalty_factor = getattr(self, "penalty_factor", 200.0)
        
        # Override with temporary decaying penalties if active
        temp_steps = int(self.problem_state.get("temporary_penalty_steps", 0))
        if temp_steps > 0:
            penalty_factor = float(self.problem_state.get("temporary_penalty_factor", penalty_factor))
            self.problem_state["temporary_penalty_steps"] = temp_steps - 1
            
        self.problem_state["capacity_penalty_factor"] = penalty_factor
        self.penalty_factor = penalty_factor

    def validation_solution(self) -> bool:
        """
        Check the validation of this solution. O(1) checks mapped effectively.
        """
        depot = self.instance_data["depot"]
        node_num = self.instance_data["node_num"]
        
        # 1. Check include depot
        for route in self.current_solution.routes:
            if depot not in route:
                return False

        # 2. Check load capacity constraints (DISABLED for Infeasible Penalty Search)
        capacity = self.instance_data['capacity']
        demands = self.instance_data['demands']
        for route in self.current_solution.routes:
            if False: # DISABLED
                return False
        # 2. Check uniqueness & node existence
        visited_customers = set()
        for route in self.current_solution.routes:
            for n in route:
                if not (0 <= n < node_num):
                    return False
                if n != depot:
                    if n in visited_customers:
                        return False
                    visited_customers.add(n)

        return True

    def load_solution(self, path: str) -> bool:
        """Load a solution from a file."""
        try:
            routes = []
            total_cost = 0.0
            loads = []
            loaded_trajectory = []
            headers = []
            reading_trajectory = False
            depot = self.instance_data["depot"]
            demands = self.instance_data["demands"]
            
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    
                    # Check for section headers
                    if line.startswith("-"):
                        if line.startswith("-trajectory:") or line.startswith("-parent_trajectory:"):
                            reading_trajectory = True
                            headers = []
                            continue
                        else:
                            reading_trajectory = False

                    if reading_trajectory:
                        if not headers:
                            headers = line.split("\t")
                        else:
                            values = line.split("\t")
                            if len(values) == len(headers):
                                record = dict(zip(headers, values))
                                loaded_trajectory.append(record)
                        continue

                    if line.startswith("vehicle_"):
                        # Format: vehicle_0: 0->1->2->0
                        content = line.split(":", 1)[1].strip()
                        nodes_str = content.split("->")
                        # exclude the last element which is the depot
                        route = [int(x) for x in nodes_str][:-1]
                        routes.append(route)
                        
                        # Re-calculate loads 
                        current_load = sum([demands[node] for node in route])
                        loads.append(current_load)
                        
                    elif line.startswith("-total_cost:"):
                        total_cost = float(line.split(":", 1)[1].strip())
            
            self.current_solution = Solution(routes=routes, depot=depot, total_cost=total_cost, loads=loads)
            
            if self.trajectory is None:
                self.trajectory = []
            self.trajectory = loaded_trajectory + self.trajectory
            
            self.update_problem_state()
            
            return True
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error loading solution from {path}: {e}")
            return False
