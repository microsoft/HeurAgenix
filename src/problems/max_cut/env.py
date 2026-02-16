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
            # Skip comments
            line = file.readline()
            while line and line.strip().startswith("#"):
                line = file.readline()
                
            node_num = int(line.split(" ", 1)[0])
            weight_matrix = np.zeros((node_num, node_num))
            adj = [{} for _ in range(node_num)]
            for row in file:
                if row.strip().startswith("#"): continue
                parts = row.strip("\n").split()
                if len(parts) < 3: continue
                # Modified to support float weights for MQLib instances
                node_1 = int(parts[0])
                node_2 = int(parts[1])
                weight = float(parts[2])
                
                # Adjust for 1-based indexing if needed, usually datasets are 1-based
                # Check bounds casually?
                if node_1 > node_num or node_2 > node_num:
                     pass # Should warn or handle 0-based? Assuming 1-based as per Gset/MQLib standard
                weight_matrix[node_1 - 1][node_2 - 1] = weight
                weight_matrix[node_2 - 1][node_1 - 1] = weight
                adj[node_1 - 1][node_2 - 1] = weight
                adj[node_2 - 1][node_1 - 1] = weight
        
        # Calculate data scale for adaptive algorithm parameters
        nonzero_weights = weight_matrix[weight_matrix != 0]
        if len(nonzero_weights) > 0:
            self.mean_weight = float(np.mean(np.abs(nonzero_weights)))
        else:
            self.mean_weight = 1.0

        return {"node_num": node_num, "weight_matrix": weight_matrix, "adj": adj}
    
    def reset(self, output_dir: str=None):
        super().reset(output_dir)
        # Adaptive Temperature Initialization for Large Scale Weights (ImgSeg)
        # Standard Gset (Weight~1) uses T=100. scaling_factor = 100.
        # UPDATE: Increased to 1000.0 to break stagnation in large float graphs (2026-02-10)
        base_scaling = 1000.0
        adaptive_temp = self.mean_weight * base_scaling
        
        # Inject into algorithm_data so heuristics pick it up automatically
        self.algorithm_data["temperature"] = adaptive_temp
        self.algorithm_data["initial_temperature"] = adaptive_temp
        # Also scale final_temperature if heuristics use it
        self.algorithm_data["final_temperature"] = self.algorithm_data.get("final_temperature", 0.001) * self.mean_weight

    def init_solution(self) -> Solution:
        return Solution(set_a=set(), set_b=set(), cut_value=0)

    def get_key_value(self, solution: Solution=None) -> float:
        """Get the key value of the current solution based on the key item."""
        # Optimization: Trust self.current_solution.cut_value if available and no specific solution passed
        if solution is None:
            if self.current_solution.cut_value is not None:
                return self.current_solution.cut_value
            solution = self.current_solution
        
        # If a specific solution is passed, or current solution has no cut_value, calculate it
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
            # Check for potential overlap/re-assignment (Fix for Cosm/Batch heuristics on assigned nodes)
            # If nodes are already assigned, we must use full recalculation or complex delta logic.
            # We use full recalculation if any overlap is detected (safer).
            overlap_detected = False
            # Quick sample check (checking first 10 nodes)
            check_nodes = (operator.nodes_to_a[:10] if operator.nodes_to_a else []) + \
                        (operator.nodes_to_b[:10] if operator.nodes_to_b else [])
            for u in check_nodes:
                if u in solution.set_a or u in solution.set_b:
                    overlap_detected = True
                    break
            
            if overlap_detected:
                # FULL DELTA CALCULATION (Robust Mode)
                # Calculates delta = New_Cut - Old_Cut
                
                # Virtual Sets
                temp_set_a = solution.set_a.copy()
                temp_set_b = solution.set_b.copy()
                
                if operator.nodes_to_a:
                    temp_set_a.update(operator.nodes_to_a)
                    temp_set_b.difference_update(operator.nodes_to_a)
                if operator.nodes_to_b:
                    temp_set_b.update(operator.nodes_to_b)
                    temp_set_a.difference_update(operator.nodes_to_b)
                
                new_cut_value = 0
                wm = self.instance_data["weight_matrix"]
                
                # Calculate New Cut
                # Optimization: Iterate smaller set
                if len(temp_set_a) < len(temp_set_b):
                    for u in temp_set_a:
                        for v in temp_set_b:
                             new_cut_value += wm[u][v]
                else:
                    for u in temp_set_b:
                        for v in temp_set_a:
                             new_cut_value += wm[u][v] # Symmetric
                             
                delta = new_cut_value - (solution.cut_value if solution.cut_value else 0)

            else:
                # FAST INCREMENTAL DELTA (Assumes Disjoint / New Nodes)
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
        """
        Apply the operator to the current solution In-Place.
        This updates self.current_solution directly without creating a new Solution object.
        """
        if isinstance(operator, BaseOperator):
            delta = self._calculate_delta(operator)
            
            # IN-PLACE UPDATE LOGIC
            solution = self.current_solution
            
            if isinstance(operator, InsertNodeOperator):
                if operator.target_set == "A":
                    solution.set_a.add(operator.node)
                    # operator.run asserts node not in B, but for speed we might skip or trust input
                    # If we want to be safe:
                    if operator.node in solution.set_b: solution.set_b.remove(operator.node)
                elif operator.target_set == "B":
                    solution.set_b.add(operator.node)
                    if operator.node in solution.set_a: solution.set_a.remove(operator.node)
                    
            elif isinstance(operator, InsertEdgeOperator):
                # node_1 to A, node_2 to B
                solution.set_a.add(operator.node_1)
                if operator.node_1 in solution.set_b: solution.set_b.remove(operator.node_1)
                
                solution.set_b.add(operator.node_2)
                if operator.node_2 in solution.set_a: solution.set_a.remove(operator.node_2)
                
            elif isinstance(operator, SwapOperator):
                for node in operator.nodes:
                    if node in solution.set_a:
                        solution.set_a.remove(node)
                        solution.set_b.add(node)
                    elif node in solution.set_b:
                        solution.set_b.remove(node)
                        solution.set_a.add(node)

            elif isinstance(operator, DeleteOperator):
                u = operator.node
                if u in solution.set_a: solution.set_a.remove(u)
                if u in solution.set_b: solution.set_b.remove(u)
            
            elif isinstance(operator, BatchInsertNodeOperator):
                # nodes_to_a
                if operator.nodes_to_a:
                    solution.set_a.update(operator.nodes_to_a)
                    solution.set_b.difference_update(operator.nodes_to_a)
                # nodes_to_b
                if operator.nodes_to_b:
                    solution.set_b.update(operator.nodes_to_b)
                    solution.set_a.difference_update(operator.nodes_to_b)

            elif isinstance(operator, BatchDeleteOperator):
                if operator.nodes:
                    solution.set_a.difference_update(operator.nodes)
                    solution.set_b.difference_update(operator.nodes)
            
            # --- End In-Place Update ---

            # Update cut_value
            if solution.cut_value is not None and delta != 0:
                solution.cut_value += delta
            elif solution.cut_value is None:
                # If delta calculation failed or it's a fresh start
                solution.cut_value = self.get_key_value(solution)

            self.problem_state = self.get_problem_state()
            return True # Indicate success
        return False

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

    def dump_best_solution(self, path: str) -> None:
        """Dump the current best solution to a file."""
        # Temporarily swap output_dir to dump to the specific path
        original_output_dir = self.output_dir
        
        try:
            # path is like ".../high_quality_solution/current_best.12345.exp.runid"
            # dump_result expects a directory and a filename
            target_dir = os.path.dirname(path)
            target_file = os.path.basename(path)
            
            self.output_dir = target_dir
            
            # Use dump_result to get full info (trajectory, etc.)
            # We use a temp file first for atomic write safety
            temp_file = target_file + ".tmp"
            self.dump_result(result_file=temp_file)
            
            # Atomic rename
            temp_path = os.path.join(target_dir, temp_file)
            final_path = os.path.join(target_dir, target_file)
            os.replace(temp_path, final_path)
            
        except Exception as e:
            print(f"Error dumping solution to {path}: {e}")
        finally:
            # Restore original output_dir
            self.output_dir = original_output_dir

    def load_solution(self, path: str) -> bool:
        """Load a solution from a file."""
        try:
            set_a = set()
            set_b = set()
            cut_value = 0.0
            loaded_recordings = []
            headers = []
            reading_trajectory = False
            
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
                            # Other sections (e.g. -data, -current_solution)
                            reading_trajectory = False
                            # Fall through to check specific fields if needed, 
                            # but usually fields like set_a don't start with -
                            pass

                    if reading_trajectory:
                        if not headers:
                            headers = line.split("\t")
                        else:
                            values = line.split("\t")
                            if len(values) == len(headers):
                                record = dict(zip(headers, values))
                                loaded_recordings.append(record)
                        continue

                    if line.startswith("set_a:"):
                        content = line.split(":", 1)[1].strip()
                        if content:
                            set_a = {int(x) - 1 for x in content.split(",")}
                    elif line.startswith("set_b:"):
                        content = line.split(":", 1)[1].strip()
                        if content:
                            set_b = {int(x) - 1 for x in content.split(",")}
                    elif line.startswith("cut_value:"):
                        cut_value = float(line.split(":", 1)[1].strip())
            
            self.current_solution = Solution(set_a=set_a, set_b=set_b, cut_value=cut_value)
            
            # Append loaded recordings to self.recordings
            if self.recordings is None:
                self.recordings = []
            # Prepend loaded recordings to maintain history
            self.recordings = loaded_recordings + self.recordings
            
            # Update problem state
            self.problem_state = self.get_problem_state()
            
            return True
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error loading solution from {path}: {e}")
            return False
