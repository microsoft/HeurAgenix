import os
import traceback
import time
from src.problems.base.components import BaseSolution, BaseOperator


class BaseEnv:
    """Base env that stores the static global data, current solution, dynamic state and provide necessary to support algorithm."""
    def __init__(self, data_name: str, mode: str, problem: str, **kwargs):
        self.data_name = data_name
        self.mode = mode
        self.problem = problem
        assert mode in ["train", "validation", "test", "smoke"]
        if os.path.exists(data_name):
            self.data_path = data_name
        elif os.path.exists(os.path.join("src", "problems", problem, "data", f"{self.mode}_data", data_name)):
            self.data_path = os.path.join("src", "problems", problem, "data", f"{self.mode}_data", data_name)
        elif os.path.exists(os.path.join("output", problem, f"{self.mode}_data", data_name)):
            self.data_path = os.path.join(os.path.join("output", problem, f"{self.mode}_data", data_name))
        self.data: tuple = self.load_data(self.data_path)
        self.current_solution: BaseSolution = None
        self.global_data: dict = None
        self.state_data: dict = None
        self.algorithm_data: dict = None
        self.recording: list[tuple] = None
        self.output_dir: str = None
        self.time_cost: float = 0
        # Maximum step to constructive a complete solution
        self.construction_steps: int = None
        # Key item in state to compare the solution
        self.key_item: str = None
        # Returns the advantage of the first and second key value 
        # A return value greater than 0 indicates that first is better and the larger the number, the greater the advantage.
        self.compare: callable = None

    @property
    def is_complete_solution(self) -> bool:
        pass

    @property
    def key_value(self) -> float:
        return self.state_data[self.key_item]

    def reset(self, experiment_name: str=None, mode: str=None):
        self.current_solution = self.init_solution()
        self.global_data = self.get_global_data()
        self.state_data = self.get_state_data()
        self.algorithm_data = {}
        self.recording = []
        self.time_cost = 0
        if experiment_name:
            self.output_dir = os.path.join("output", self.problem, f"{self.mode}_result", self.data_name.split(os.sep)[-1], experiment_name)
            os.makedirs(self.output_dir, exist_ok=True)
        if mode:
            self.mode = mode

    def load_data(self, data_path: str) -> None:
        pass

    def init_solution(self) -> None:
        pass

    def get_global_data(self) -> dict:
        """Retrieve the global static information data as a dictionary.

        Returns:
            dict: A dictionary containing the global static information data.
        """
        pass

    def get_state_data(self) -> dict:
        """Retrieve the current dynamic state data as a dictionary.

        Returns:
            dict: A dictionary containing the current dynamic state data.
        """
        pass

    def validation_solution(self, solution: BaseSolution) -> bool:
        """Check the validation of this solution"""
        pass

    def run_heuristic(self, heuristic: callable, parameters:dict={}, inplace: bool=True, validation: bool=True) -> BaseOperator:
        try:
            start_time = time.time()
            operator, delta = heuristic(
                global_data=self.global_data,
                state_data=self.state_data,
                algorithm_data=self.algorithm_data,
                get_state_data_function=self.get_state_data,
                **parameters
            )
            end_time = time.time()
            if operator is not None:
                validation_result = self.run_operator(operator, inplace, validation)
                if validation_result:
                    self.algorithm_data.update(delta)
                    self.time_cost += end_time - start_time
                    return operator
            return False
        except Exception as e:
            trace_string = traceback.format_exc()
            print(trace_string)
            return trace_string

    def run_operator(self, operator: BaseOperator, inplace: bool=True, validation: bool=True) -> bool:
        if isinstance(operator, BaseOperator):
            solution = operator.run(self.current_solution)
            if not validation or self.validation_solution(solution):
                if inplace:
                    self.current_solution = solution
                    self.recording.append((operator, str(solution)))
                self.state_data = self.get_state_data()
                return True
        return False

    def get_observation(self) -> dict:
        pass

    def dump_result(self, content_dict: dict={}) -> str:
        content = f"-data: {self.data_name}\n"
        for item, value in content_dict.items():
            content += f"-{item}: {value}\n"
        content += f"-current_solution:\n{self.current_solution}\n"
        content += f"-is_complete_solution: {self.is_complete_solution}\n"
        if self.mode == "train":
            trajectory_str = "\n".join([
                str(index) + "\t" + str(operator) + "\t" + solution_str.replace("\n", r"\n")
                for index, (operator, solution_str) in enumerate(self.recording)
            ])
            content += f"-trajectory:\noperation_id\toperator(parameter)\tsolution_after_operation\n{trajectory_str}\n"

        if self.output_dir != None:
            output_file = os.path.join(self.output_dir, "result.txt")
            with open(output_file, "w") as file:  
                file.write(content) 
        
        return content