from src.problems.base.env import BaseEnv
# Import Solution from problems
# For example for tsp
# from src.problems.tsp.components import Solution


class Env(BaseEnv):
    """xxx env that stores the static global data, current solution, dynamic state and provide necessary support to the algorithm."""
    def __init__(self, data_name: str, **kwargs):
        # 1. Init the Env. xxx is the problem name.
        super().__init__(data_name, "xxx")

        # 2. Get the data by . The data should align with output from function def load_data.
        self.data1, self.data2, ... = self.data

        # 3. Clarity the key item that indicates the performance. xxx is the key item in state_data (defined in get_state_data function)
        self.key_item = "xxx"

        # 4. Clarity the comparison function. (higher is better of lower is better)
        # self.compare = lambda x, y: y - x # lower is better
        self.compare = lambda x, y: x - y # higher is better

        # 5. Clarity the construction steps for the (Optional)
        self.construction_steps = ...

        # Other necessary varibles/state_data for special problem

        # For example, for tsp:
        # super().__init__(data_name, "tsp")
        # self.node_num, self.distance_matrix = self.data
        # self.construction_steps = self.node_num
        # self.key_item = "current_cost"
        # self.compare = lambda x, y: y - x
    

    def load_data(self, data_path: str) -> None:
        # load data and return them.
        pass

        # For example, for tsp:
        # problem = tsplib95.load(data_path)
        # distance_matrix = nx.to_numpy_array(problem.get_graph())
        # node_num = len(distance_matrix)
        # return node_num, distance_matrix


    @property
    def is_complete_solution(self) -> bool:
        # Check whether the solution is complete.
        # For some problems, we have hard indicators to determine whether the solution is complete, so we need to determine whether the current solution is complete.
        # For example, for the TSP problem, we need to visit all nodes
        # return len(set(self.current_solution.tour)) == self.node_num
        # For some problem we do not care about or cannot judge the completeness of the solution, just return True.
        # For example, for the knapsack problem, we do not need to worry about completeness, because it is possible that not even one item can be loaded. 
        pass

    def init_solution(self) -> None:
        # Init the solution for special problem
        # For example, for TSP
        # return Solution(tour=[])
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

    def get_global_data(self) -> dict:
        """Retrieve the global static information data as a dictionary.

        Returns:
            dict: A dictionary containing the global static information data with:
                - "xx" (xx type): xxxx
                - "xx" (xx type): xxxx
        """

        global_data_dict = {
            # add global data here
        }
        return global_data_dict

    def get_state_data(self, solution=None) -> dict:
        """Retrieve the current dynamic state data as a dictionary.

        Returns:
            dict: A dictionary containing the current dynamic state data with:
                - "current_solution" (Solution): An instance of the Solution class representing the current solution.
                - "validation_solution" (callable): def validation_solution(solution: Solution) -> bool: function to check whether new solution is valid.
                - "xx" (xx type): xxxx
                - "xx" (xx type): xxxx
        """
        if solution is None:
            solution = self.current_solution

        state_data_dict = {
            "current_solution": solution,
            "validation_solution": self.validation_solution,
            # add other state data here
        }
        return state_data_dict

    def dump_result(self, dump_trajectory: bool=True) -> str:
        content_dict = {
            # The data name, current solution, complete flag, validation flag and trajectory will be added by super().dump_result(content_dict)
            # add other information to dump.
        }
        content = super().dump_result(content_dict)
        return content
