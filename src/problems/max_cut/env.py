import os
import numpy as np
from src.problems.base.env import BaseEnv
from src.problems.base.components import BaseOperator
from src.problems.max_cut.components import Solution, InsertNodeOperator, InsertEdgeOperator, SwapOperator, DeleteOperator, BatchInsertNodeOperator, BatchDeleteOperator
from src.problems.max_cut.best_known import best_known


class Env(BaseEnv):
    """MaxCut env that stores the instance data, current solution, and problem state to support algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "max_cut")
        self.construction_steps = self.instance_data["node_num"]
        self.key_item = "current_cut_value"
        self.compare = lambda x, y: x - y

    @property
    def is_complete_solution(self) -> bool:
        return len(self.current_solution.set_a) + len(self.current_solution.set_b) == self.instance_data["node_num"]

    def load_data(self, data_path: str) -> tuple:
        data_name = data_path.split(os.sep)[-1].split(".")[0]
        self.best_known = best_known.get(data_name, None)
        with open(data_path) as file:
            node_num = int(file.readline().split(" ", 1)[0])
            weight_matrix = np.zeros((node_num, node_num))
            adj = [{} for _ in range(node_num)]
            for row in file:
                node_1, node_2, weight = [int(e) for e in row.strip("\n").split()]
                weight_matrix[node_1 - 1][node_2 - 1] = weight
                weight_matrix[node_2 - 1][node_1 - 1] = weight
                adj[node_1 - 1][node_2 - 1] = weight
                adj[node_2 - 1][node_1 - 1] = weight
        return {"node_num": node_num, "weight_matrix": weight_matrix, "adj": adj}

    def init_solution(self) -> Solution:
        return Solution(set_a=set(), set_b=set(), cut_value=0)

    def get_key_value(self, solution: Solution=None) -> float:
        """Get the key value of the current solution based on the key item."""
        if solution is None:
            solution = self.current_solution
        
        if solution.cut_value is not None:
            return solution.cut_value

        current_cut_value = 0
        for node_a in solution.set_a:
            for node_b in solution.set_b:
                current_cut_value += self.instance_data["weight_matrix"][node_a][node_b]
        return current_cut_value

    def _calculate_delta(self, operator: BaseOperator) -> float:
        adj = self.instance_data["adj"]
        solution = self.current_solution
        delta = 0
        
        if isinstance(operator, InsertNodeOperator):
            u = operator.node
            target = operator.target_set
            neighbors = adj[u]
            if target == "A":
                for v, w in neighbors.items():
                    if v in solution.set_b:
                        delta += w
            else: # target == "B"
                for v, w in neighbors.items():
                    if v in solution.set_a:
                        delta += w
                        
        elif isinstance(operator, InsertEdgeOperator):
            u, v = operator.node_1, operator.node_2
            # u -> A, v -> B
            for neighbor, w in adj[u].items():
                if neighbor in solution.set_b:
                    delta += w
            for neighbor, w in adj[v].items():
                if neighbor in solution.set_a:
                    delta += w
            if v in adj[u]:
                delta += adj[u][v]
                
        elif isinstance(operator, DeleteOperator):
            u = operator.node
            if u in solution.set_a:
                for v, w in adj[u].items():
                    if v in solution.set_b:
                        delta -= w
            elif u in solution.set_b:
                for v, w in adj[u].items():
                    if v in solution.set_a:
                        delta -= w

        elif isinstance(operator, SwapOperator):
            nodes = set(operator.nodes)
            # 1. Individual contributions
            for u in nodes:
                neighbors = adj[u]
                if u in solution.set_a:
                    # A -> B
                    for v, w in neighbors.items():
                        if v in solution.set_a:
                            delta += w
                        elif v in solution.set_b:
                            delta -= w
                elif u in solution.set_b:
                    # B -> A
                    for v, w in neighbors.items():
                        if v in solution.set_b:
                            delta += w
                        elif v in solution.set_a:
                            delta -= w
            
            # 2. Correction for internal edges within the moving set
            sorted_nodes = sorted(list(nodes))
            for i in range(len(sorted_nodes)):
                u = sorted_nodes[i]
                u_in_a = u in solution.set_a
                for v, w in adj[u].items():
                    if v in nodes and v > u:
                        v_in_a = v in solution.set_a
                        
                        if u_in_a and v_in_a:
                            delta -= 2 * w
                        elif not u_in_a and not v_in_a:
                            delta -= 2 * w
                        else:
                            delta += 2 * w
                            
        elif isinstance(operator, BatchInsertNodeOperator):
            nodes_to_a = operator.nodes_to_a
            nodes_to_b = operator.nodes_to_b
            
            # 1. Edges between New A and Existing B
            for u in nodes_to_a:
                for v, w in adj[u].items():
                    if v in solution.set_b:
                        delta += w
                        
            # 2. Edges between New B and Existing A
            for u in nodes_to_b:
                for v, w in adj[u].items():
                    if v in solution.set_a:
                        delta += w
                        
            # 3. Edges between New A and New B
            set_nodes_to_b = set(nodes_to_b)
            for u in nodes_to_a:
                for v, w in adj[u].items():
                    if v in set_nodes_to_b:
                        delta += w

        elif isinstance(operator, BatchDeleteOperator):
            for u in operator.nodes:
                if u in solution.set_a:
                    for v, w in adj[u].items():
                        if v in solution.set_b:
                            delta -= w
                elif u in solution.set_b:
                    for v, w in adj[u].items():
                        if v in solution.set_a:
                            delta -= w
                
        return delta

    def run_operator(self, operator: BaseOperator) -> bool:
        if isinstance(operator, BaseOperator):
            delta = self._calculate_delta(operator)
            self.current_solution = operator.run(self.current_solution)
            
            if self.current_solution.cut_value is None:
                 # If previous solution had cut_value, we can update it.
                 # But operator.run returns a new Solution with cut_value copied from old solution (based on my change to components.py)
                 # Wait, I modified components.py to copy cut_value.
                 # So self.current_solution.cut_value should be the OLD value.
                 pass
            
            if self.current_solution.cut_value is not None:
                self.current_solution.cut_value += delta
            else:
                # Fallback if somehow it's None (e.g. first run or something)
                self.current_solution.cut_value = self.get_key_value(self.current_solution)

            self.problem_state = self.get_problem_state()
        return operator

    def validation_solution(self, solution: Solution=None) -> bool:
        """Check the validation of this solution in the following items:
            1. Non-repeat: No nodes in both set A and set B
        """
        if solution is None:
            solution = self.current_solution

        if not isinstance(solution, Solution) or not isinstance(solution.set_a, set) or not isinstance(solution.set_b, set):
            return False

        # Check non-repeat
        all_selected_nodes = solution.set_a.union(solution.set_b)
        if len(all_selected_nodes) != len(solution.set_a) + len(solution.set_b):
            return False

        return True
