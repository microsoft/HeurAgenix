import os
import traceback
import time
from src.problems.base.components import BaseSolution, BaseOperator
from src.util.util import search_file


class BaseEnv:
    """Base env that stores the static global data, current solution, dynamic state and provide necessary to support algorithm."""
    def __init__(self, data_name: str, problem: str, **kwargs):
        self.data_name = data_name
        self.problem = problem
        self.online_problem = False
        self.data_path = search_file(data_name, problem)
        assert self.data_path is not None
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
    def is_valid_solution(self) -> bool:
        return self.validation_solution(self.current_solution)

    @property
    def continue_run(self) -> bool:
        return True

    @property
    def key_value(self) -> float:
        return self.state_data[self.key_item]

    def reset(self, experiment_name: str=None):
        self.current_solution = self.init_solution()
        self.global_data = self.get_global_data()
        self.state_data = self.get_state_data()
        self.algorithm_data = {}
        self.recording = []
        self.time_cost = 0
        if experiment_name:
            if os.sep in experiment_name:
                self.output_dir = experiment_name
            else:
                self.output_dir = os.path.join("output", self.problem, "result", self.data_name.split(os.sep)[-1], experiment_name)
            os.makedirs(self.output_dir, exist_ok=True)

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

    def validation_solution(self, solution: BaseSolution=None) -> bool:
        """Check the validation of this solution"""
        pass

    def run_heuristic(self, heuristic: callable, parameters:dict={}, inplace: bool=True) -> BaseOperator:
        #try:
        if True:
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
                result = self.run_operator(operator, inplace, heuristic.__name__)
                if result:
                    self.algorithm_data.update(delta)
                    self.time_cost += end_time - start_time
                    return operator
            return None
        else:
        # except Exception as e:
            trace_string = traceback.format_exc()
            print(trace_string)
            return trace_string

    def run_operator(self, operator: BaseOperator, inplace: bool=True, heuristic_name: str=None) -> bool:
        if isinstance(operator, BaseOperator):
            solution = operator.run(self.current_solution)
            if inplace:
                self.current_solution = solution
                self.recording.append((str(heuristic_name), operator, str(solution)))
            self.state_data = self.get_state_data()
            return operator
        return None

    def get_observation(self) -> dict:
        pass

    def summarize_env(self) -> str:
        pass

    def dump_result(self, content_dict: dict={}, dump_trajectory: bool=True, compress_trajectory: bool=False, result_file: str="result.txt") -> str:
        content = f"-data: {self.data_name}\n"
        content += f"-current_solution:\n{self.current_solution}\n"
        content += f"-is_complete_solution: {self.is_complete_solution}\n"
        content += f"-is_valid_solution: {self.is_valid_solution}\n"
        content += f"-{self.key_item}: {self.key_value}\n"
        for item, value in content_dict.items():
            content += f"-{item}: {value}\n"
        if dump_trajectory:
            if compress_trajectory:
                trajectory_str = "\n".join([
                    str(index) + "\t" + str(operator)
                    for index, (heuristic_name, operator, solution_str) in enumerate(self.recording)
                ])
                content += f"-trajectory:\noperation_id\toperator(parameter)\n{trajectory_str}\n"
            else:
                trajectory_str = "\n".join([
                    str(index) + "\t" + heuristic_name + "\t" + str(operator) + "\t" + solution_str.replace("\n", r"\n")
                    for index, (heuristic_name, operator, solution_str) in enumerate(self.recording)
                ])
                content += f"-trajectory:\noperation_id\theuristic\toperator(parameter)\tsolution_after_operation\n{trajectory_str}\n"

        if self.output_dir != None:
            output_file = os.path.join(self.output_dir, result_file)
            with open(output_file, "w") as file:  
                file.write(content) 
        
        return content
