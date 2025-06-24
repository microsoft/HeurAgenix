from src.problems.base.env import BaseEnv
from src.problems.base.components import BaseSolution


class Env(BaseEnv):
    """xxx env that stores the instance data, current solution, and problem state to support algorithm."""
    def __init__(self, data_name: str, **kwargs):
        # 1. Init the Env. xxx is the problem name.
        super().__init__(data_name, "xxx")

        # 2. Clarity the key item that indicates the performance. (calculated by get_key_value)
        self.key_item = "xxx"

        # 3. Clarity the comparison function. (higher is better of lower is better)
        # self.compare = lambda x, y: y - x # lower is better
        self.compare = lambda x, y: x - y # higher is better

        # 4. Clarity the construction steps for the (Optional)
        self.construction_steps = ...

        # Other necessary logic for special problem

        # For example, for tsp:
        # super().__init__(data_name, "tsp")
        # self.construction_steps = self.node_num
        # self.key_item = "current_cost"
        # self.compare = lambda x, y: y - x
        # The data load by load_data can be used as self.instance_data[xxx]
        # Such as self.instance_data["node_num"] and self.instance_data["distance_matrix"]
    

    def load_data(self, data_path: str) -> dict:
        # load data and return by dict.
        pass

        # For example, for tsp:
        # problem = tsplib95.load(data_path)
        # distance_matrix = nx.to_numpy_array(problem.get_graph())
        # node_num = len(distance_matrix)
        # return {"node_num": node_num, "distance_matrix": distance_matrix}


    @property
    def is_complete_solution(self) -> bool:
        # Check whether the solution is complete.
        # For some problems, we have hard indicators to determine whether the solution is complete, so we need to determine whether the current solution is complete.
        # For example, for the TSP problem, we need to visit all nodes
        # return len(set(self.current_solution.tour)) == self.node_num
        # For some problem we do not care about or cannot judge the completeness of the solution, just return True.
        pass

    def init_solution(self) -> None:
        # Init the solution for special problem
        # For example, for TSP
        # return Solution(tour=[])
        pass

    def get_key_value(self, solution: BaseSolution=None):
        # Get the value of key item that indicates the performance.
        pass

    def validation_solution(self, solution=None) -> bool:
        """Check the validation of this solution in following items:
            1. xxx
            2. xxx
            return True if legal, otherwise False
        """
        # Default the check current_solution, but we can still check other solutions.
        if solution is None:
            solution = self.current_solution
        
        # Here we just check the validation for solution, but not complete, so partial feasible solution is also legal(True).
        # For example, for 5-node tsp problem, solution = [1, 2, 3] is legal, while [1, 2, 2] is illegal.
        # For tsp, we need to check:
        #   1. Node Existence: Each node in the solution must exist within the problem instance's range of nodes.
        #   2. Uniqueness: No node is repeated within the solution path, ensuring that each node is visited at most once.
        #   3. Connectivity: Each edge (from one node to the next) must be connected, i.e., not marked as infinite distance in the distance matrix.

        # Implement the validation code here.
        pass

    def dump_result(self, content_dict: dict={}, dump_records: list=["operation_id", "operator", "heuristic"], result_file: str="result.txt") -> str:
        # Optional: Dump the result of the env.
        # Instance data, solution, key item, trajectory with heuristic and operation are dumped by default.
        content_dict = {
            # The data name, current solution, complete flag, validation flag and trajectory will be added by super().dump_result(content_dict)
            # add other information to dump.
        }
        content = super().dump_result(content_dict=content, dump_records=dump_records, result_file=result_file)
        return content
