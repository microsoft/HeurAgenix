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
                        distance_matrix[i][j] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
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
        return Solution(routes=[[self.instance_data["depot"]] for _ in range(self.instance_data["vehicle_num"])], depot=self.instance_data["depot"])

    def get_key_value(self, recalculate: bool=False) -> float:
        """Get the key value of the current solution based on the key item."""
        if not recalculate and self.current_solution.total_cost is not None:
            return self.current_solution.total_cost
        
        solution = self.current_solution
        total_current_cost = 0
        for vehicle_index in range(self.instance_data["vehicle_num"]):
            route = solution.routes[vehicle_index]
            # The cost of the current solution for each vehicle.
            if len(route) == 0: continue
            
            # Start from depot
            cost_for_vehicle = self.instance_data["distance_matrix"][self.instance_data["depot"]][route[0]]
            
            cost_for_vehicle += sum([self.instance_data["distance_matrix"][route[index]][route[index + 1]] for index in range(len(route) - 1)])
            
            # Back to depot
            cost_for_vehicle += self.instance_data["distance_matrix"][route[-1]][self.instance_data["depot"]]
            
            total_current_cost += cost_for_vehicle
        return total_current_cost

    def _get_route_cost(self, route: list[int]) -> float:
        if not route:
            return 0.0
        depot = self.instance_data["depot"]
        cost = self.instance_data["distance_matrix"][depot][route[0]]
        cost += sum([self.instance_data["distance_matrix"][route[i]][route[i+1]] for i in range(len(route)-1)])
        cost += self.instance_data["distance_matrix"][route[-1]][depot]
        return cost

    def _calculate_delta(self, operator: BaseOperator) -> float:
        if not isinstance(operator, BaseOperator):
            return 0.0
            
        dist = self.instance_data["distance_matrix"]
        depot = self.instance_data["depot"]
        solution = self.current_solution
        delta = 0.0

        if isinstance(operator, AppendOperator):
            # Append node to end of route
            # Old: ... -> last -> depot
            # New: ... -> last -> node -> depot
            vid = operator.vehicle_id
            node = operator.node
            route = solution.routes[vid]
            
            if not route:
                # depot -> node -> depot
                # Old: 0 (empty route cost is 0 if we consider it doesn't leave depot)
                delta = dist[depot][node] + dist[node][depot]
            else:
                last = route[-1]
                # Remove last->depot, add last->node, node->depot
                delta = -dist[last][depot] + dist[last][node] + dist[node][depot]

        elif isinstance(operator, InsertOperator):
            # Insert node at position
            vid = operator.vehicle_id
            node = operator.node
            pos = operator.position
            route = solution.routes[vid]
            
            # Identify prev and next nodes
            prev_node = route[pos-1] if pos > 0 else depot
            next_node = route[pos] if pos < len(route) else depot
            
            # Remove prev->next, add prev->node, node->next
            delta = -dist[prev_node][next_node] + dist[prev_node][node] + dist[node][next_node]

        elif isinstance(operator, RelocateOperator):
            # Move node from source to target
            svid, spos = operator.source_vehicle_id, operator.source_position
            tvid, tpos = operator.target_vehicle_id, operator.target_position
            
            # Calculate cost influence by simulating the move on route copies
            # This handles both inter-route and intra-route (index shift) correctly.
            if svid != tvid:
                s_route = solution.routes[svid]
                t_route = solution.routes[tvid]
                old_cost = self._get_route_cost(s_route) + self._get_route_cost(t_route)
                
                # Simulate move
                # node = s_route[spos] 
                # Be careful not to modify original route, use list slicing/comprehension
                node = s_route[spos]
                new_s_route = s_route[:spos] + s_route[spos+1:]
                new_t_route = t_route[:tpos] + [node] + t_route[tpos:]
                
                new_cost = self._get_route_cost(new_s_route) + self._get_route_cost(new_t_route)
                delta = new_cost - old_cost
            else:
                # Same vehicle
                route = solution.routes[svid]
                old_cost = self._get_route_cost(route)
                
                node = route[spos]
                # Use slicing to remove
                temp_route = route[:spos] + route[spos+1:]
                
                # Adjust target position if needed
                actual_tpos = tpos
                if spos < tpos:
                    actual_tpos -= 1
                
                # Insert
                new_route = temp_route[:actual_tpos] + [node] + temp_route[actual_tpos:]
                
                new_cost = self._get_route_cost(new_route)
                delta = new_cost - old_cost

        elif isinstance(operator, SwapOperator):
             # Swap node1 (v1, p1) and node2 (v2, p2)
             v1, p1 = operator.vehicle_id1, operator.position1
             v2, p2 = operator.vehicle_id2, operator.position2
             
             if v1 == v2 and abs(p1 - p2) == 1:
                 # Adjacent swap in same route
                 route = solution.routes[v1]
                 first = min(p1, p2)
                 # ... -> prev -> A -> B -> next ...
                 # Swap A, B
                 # ... -> prev -> B -> A -> next ...
                 prev_node = route[first-1] if first > 0 else depot
                 node_a = route[first]
                 node_b = route[first+1]
                 next_node = route[first+2] if first + 2 < len(route) else depot
                 
                 delta = -dist[prev_node][node_a] - dist[node_a][node_b] - dist[node_b][next_node] \
                         + dist[prev_node][node_b] + dist[node_b][node_a] + dist[node_a][next_node]
             else:
                 # Non-adjacent
                 r1 = solution.routes[v1]
                 n1 = r1[p1]
                 prev1 = r1[p1-1] if p1 > 0 else depot
                 next1 = r1[p1+1] if p1 < len(r1)-1 else depot
                 
                 r2 = solution.routes[v2]
                 n2 = r2[p2]
                 prev2 = r2[p2-1] if p2 > 0 else depot
                 next2 = r2[p2+1] if p2 < len(r2)-1 else depot
                 
                 delta += -dist[prev1][n1] - dist[n1][next1] + dist[prev1][n2] + dist[n2][next1]
                 delta += -dist[prev2][n2] - dist[n2][next2] + dist[prev2][n1] + dist[n1][next2]

        elif isinstance(operator, ReverseSegmentOperator):
            # 2-opt: Reverse segment in ONE route
            vid = operator.vehicle_id
            segments = operator.segments # List of (start, end)
            # Typically 2-opt reverses route[i:j+1]
            # Edges broken: (i-1, i) and (j, j+1)
            # Edges added: (i-1, j) and (i, j+1)
            route = solution.routes[vid]
            for start, end in segments:
                # Indices in route
                node_i = route[start]
                node_j = route[end]
                
                prev_i = route[start-1] if start > 0 else depot
                next_j = route[end+1] if end < len(route)-1 else depot
                
                delta += -dist[prev_i][node_i] - dist[node_j][next_j]
                delta += dist[prev_i][node_j] + dist[node_i][next_j]

        return delta

    def run_operator(self, operator: BaseOperator) -> bool:
        """Apply the operator to the current solution In-Place."""
        if not isinstance(operator, BaseOperator):
            return False
            
        solution = self.current_solution
        recalculate_cost = False
        delta = 0.0
        
        # Pre-calculation of delta for simple operators
        if isinstance(operator, (AppendOperator, InsertOperator, ReverseSegmentOperator)):
            delta = self._calculate_delta(operator)
        elif isinstance(operator, SwapOperator):
            delta = self._calculate_delta(operator)
        elif isinstance(operator, RelocateOperator):
             if operator.source_vehicle_id != operator.target_vehicle_id:
                 delta = self._calculate_delta(operator)
             else:
                 recalculate_cost = True
        else:
            recalculate_cost = True
        
        if isinstance(operator, AppendOperator):
            solution.routes[operator.vehicle_id].append(operator.node)
            
        elif isinstance(operator, InsertOperator):
            solution.routes[operator.vehicle_id].insert(operator.position, operator.node)
            
        elif isinstance(operator, SwapOperator):
            r1 = solution.routes[operator.vehicle_id1]
            r2 = solution.routes[operator.vehicle_id2]
            r1[operator.position1], r2[operator.position2] = r2[operator.position2], r1[operator.position1]
            
        elif isinstance(operator, ReverseSegmentOperator):
            r = solution.routes[operator.vehicle_id]
            for start, end in operator.segments:
                # Reverse the segment in place
                r[start:end+1] = r[start:end+1][::-1]
                
        elif isinstance(operator, RelocateOperator):
            node = solution.routes[operator.source_vehicle_id].pop(operator.source_position)
            # Adjust target position if same route and source < target
            t_pos = operator.target_position
            if operator.source_vehicle_id == operator.target_vehicle_id and operator.source_position < operator.target_position:
                 t_pos -= 1
            solution.routes[operator.target_vehicle_id].insert(t_pos, node)
            
        elif isinstance(operator, BatchRemoveOperator):
            # Remove nodes
            nodes_to_remove = set(operator.nodes)
            for i in range(len(solution.routes)):
                solution.routes[i] = [n for n in solution.routes[i] if n not in nodes_to_remove]
                
        elif isinstance(operator, BatchInsertOperator):
            # Insertions: list of (vid, pos, node)
            # Sort insertions by position descending to avoid index shifting if in same route
            # But wait, logic might be complex. simplest is to group by vehicle?
            # Assuming operator.insertions is well-formed.
            # We must handle one by one or carefully.
            
            # Group by vehicle
            inserts_by_vehicle = {}
            for vid, pos, node in operator.insertions:
                if vid not in inserts_by_vehicle: inserts_by_vehicle[vid] = []
                inserts_by_vehicle[vid].append((pos, node))
            
            for vid, ops in inserts_by_vehicle.items():
                # Sort by position descending
                 ops.sort(key=lambda x: x[0], reverse=True)
                 for pos, node in ops:
                     solution.routes[vid].insert(pos, node)
        
        elif isinstance(operator, MergeRoutesOperator):
             depot = self.instance_data["depot"]
             r_src = solution.routes[operator.source_vehicle_id]
             r_tgt = solution.routes[operator.target_vehicle_id]
             
             # Extract nodes from source (excluding depot)
             src_nodes = []
             if depot in r_src:
                 idx = r_src.index(depot)
                 src_nodes = r_src[idx+1:] + r_src[:idx]
             else:
                 src_nodes = r_src[:] # Should not happen if valid
                 
             # Extract nodes from target (excluding depot)
             tgt_nodes = []
             if depot in r_tgt:
                 idx = r_tgt.index(depot)
                 tgt_nodes = r_tgt[idx+1:] + r_tgt[:idx]
             else:
                 tgt_nodes = r_tgt[:]
                 
             # Merge: Depot + Src + Tgt
             solution.routes[operator.target_vehicle_id] = [depot] + src_nodes + tgt_nodes
             solution.routes[operator.source_vehicle_id] = [depot]


        # Update Costs and Loads
        if recalculate_cost or solution.total_cost is None:
            solution.total_cost = self.get_key_value(recalculate=True)
            # Also update loads?
            # self._update_loads(solution)
        else:
            solution.total_cost += delta
            
        # Update problem state
        self.update_problem_state()
        return True


    def validation_solution(self) -> bool:
        """
        Check the validation of this solution in following items:
            1. Node existence: Each node in each route must be within the valid range.
            2. Uniqueness: Each node (except for the depot) must only be visited once across all routes.
            3. Include depot: Each route must include at the depot.
            4. Capacity constraints: The load of each vehicle must not exceed its capacity.
        """
        # Check node existence
        for route in self.current_solution.routes:
            for node in route:
                if not (0 <= node < self.instance_data["node_num"]):
                    return False

        # Check uniqueness
        all_nodes = [node for route in self.current_solution.routes for node in route if node != self.instance_data["depot"]] + [self.instance_data["depot"]]
        if len(all_nodes) != len(set(all_nodes)):
            return False

        for route in self.current_solution.routes:
            # Check include depot
            if self.instance_data["depot"] not in route:
                return False

            # Check vehicle load capacity constraints
            load = sum(self.instance_data["demands"][node] for node in route)
            if load > self.instance_data["capacity"]:
                return False

        return True
