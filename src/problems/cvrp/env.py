import os
import tsplib95
import numpy as np
import pandas as pd
import networkx as nx
from src.problems.base.env import BaseEnv
from src.problems.base.components import BaseOperator
from src.problems.cvrp.components import Solution, AppendOperator, InsertOperator, SwapOperator, ReverseSegmentOperator, RelocateOperator, BatchRemoveOperator, BatchInsertOperator, MergeRoutesOperator
from src.problems.cvrp.best_known import best_known


class Env(BaseEnv):
    """CVRP env that stores the instance data, current solution, and problem state to support algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "cvrp")
        self.construction_steps = self.instance_data["node_num"]
        self.key_item = "total_current_cost"
        self.compare = lambda x, y: y - x

    @property
    def is_complete_solution(self) -> bool:
        return len(set([node for route in self.current_solution.routes for node in route])) == self.instance_data["node_num"]

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
        capacity = problem.capacity
        demands = np.array(list(problem.demands.values()))
        return {"node_num": node_num, "distance_matrix": distance_matrix, "depot": depot, "vehicle_num": vehicle_num, "capacity": capacity, "demands": demands}

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
        for vehicle_index in range(self.instance_data["vehicle_num"]):
            route = solution.routes[vehicle_index]
            if len(route) == 0: continue
            total_current_cost += self._get_route_cost(route)
        return total_current_cost


    def _recalculate_exact(self) -> float:
        solution = self.current_solution
        total_current_cost = 0.0
        for vehicle_index in range(self.instance_data["vehicle_num"]):
            route = solution.routes[vehicle_index]
            if len(route) == 0: continue
            total_current_cost += self._get_route_cost(route)
        return total_current_cost

    def get_key_value(self, recalculate: bool=False) -> float:
        """Get the key value of the current solution based on the key item."""
        if not recalculate and self.current_solution.total_cost is not None:
            # DEBUG: Assert exact match continuously
            # EXACT DELETED
            # ASSERT DELETED
                # RAISE DELETED
            return self.current_solution.total_cost
        
        solution = self.current_solution
        total_current_cost = 0.0
        for vehicle_index in range(self.instance_data["vehicle_num"]):
            route = solution.routes[vehicle_index]
            if len(route) == 0: continue
            total_current_cost += self._get_route_cost(route)
        return total_current_cost

    def _get_route_cost(self, route: list[int]) -> float:
        if not route:
            return 0.0
        n = len(route)
        if n == 1:
            return 0.0
        dist = self.instance_data["distance_matrix"]
        return sum([dist[route[i]][route[(i + 1) % n]] for i in range(n)])

    def _is_invalid_operator(self, operator: BaseOperator) -> bool:
        if not operator: return True
        if getattr(operator, "node", None) == self.instance_data["depot"]: return True
        if getattr(operator, "nodes", None) and self.instance_data["depot"] in operator.nodes: return True
        if isinstance(operator, InsertOperator) and operator.position == 0: return True
        if isinstance(operator, RelocateOperator) and (operator.source_position == 0 or operator.target_position == 0): return True
        if isinstance(operator, SwapOperator) and (operator.position1 == 0 or operator.position2 == 0): return True
        if isinstance(operator, ReverseSegmentOperator):
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
        if self._is_invalid_operator(operator):
            return False
            
        solution = self.current_solution
        recalculate_cost = False
        recalculate_load = False
        delta = 0.0
        
        recalculate_cost = True
        recalculate_load = True
        delta = 0.0
            
        demands = self.instance_data["demands"]
        
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
            for start, end in operator.segments:
                r[start:end+1] = r[start:end+1][::-1]
                
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
            for i in range(len(solution.routes)):
                solution.routes[i] = [n for n in solution.routes[i] if n not in nodes_to_remove]
                
        elif isinstance(operator, BatchInsertOperator):
            inserts_by_vehicle = {}
            for vid, pos, node in operator.insertions:
                if vid not in inserts_by_vehicle: inserts_by_vehicle[vid] = []
                inserts_by_vehicle[vid].append((pos, node))
            
            for vid, ops in inserts_by_vehicle.items():
                 ops.sort(key=lambda x: x[0], reverse=True)
                 for pos, node in ops:
                     solution.routes[vid].insert(pos, node)
        
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

        if recalculate_load:
            self._update_loads_full(solution)

        if recalculate_cost or solution.total_cost is None:
            solution.total_cost = self.get_key_value(recalculate=True)
        else:
            solution.total_cost += delta
            
        self.update_problem_state()
        return True

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

        # 2. Check load capacity constraints (O(N) vs old recalculate)
        capacity = self.instance_data["capacity"]
        if self.current_solution.loads is not None:
             for load in self.current_solution.loads:
                 if load > capacity:
                     return False
        else:
            for route in self.current_solution.routes:
                if sum(self.instance_data["demands"][n] for n in route) > capacity:
                    return False

        # 3. Check uniqueness & node existence
        all_nodes = []
        for route in self.current_solution.routes:
            for n in route:
                if not (0 <= n < node_num):
                    return False
                if n != depot:
                    all_nodes.append(n)
                    
        all_nodes.append(depot)
        if len(all_nodes) != len(set(all_nodes)):
            return False

        return True
