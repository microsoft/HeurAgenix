import os
import traceback
from src.problems.base.components import BaseSolution, BaseOperator
from src.util.util import load_heuristic, search_file


class BaseEnv:
    """Base env that stores the static global data, current solution, dynamic state and provide necessary to support algorithm."""
    def __init__(self, data_name: str, problem: str, **kwargs):
        self.problem = problem
        self.data_path = search_file(data_name, problem)
        self.data_ref_name = data_name.split(os.sep)[-1]
        assert self.data_path is not None
        self.instance_data: tuple = self.load_data(self.data_path)
        self.current_solution: BaseSolution = self.init_solution()
        self.algorithm_data: dict = None
        self.recording: list[tuple] = None
        self.output_dir: str = None
        # Maximum step to constructive a complete solution
        self.construction_steps: int = None
        # Key item in state to compare the solution
        self.key_item: str = None
        # Returns the advantage of the first and second key value 
        # A return value greater than 0 indicates that first is better and the larger the number, the greater the advantage.
        self.compare: callable = None

        self.instance_problem_state_function = load_heuristic("problem_state.py", problem=self.problem, function_name="get_instance_problem_state")
        self.solution_problem_state_function = load_heuristic("problem_state.py", problem=self.problem, function_name="get_solution_problem_state")
        self.problem_state = self.get_problem_state()

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
        """Get the key value of the current solution."""
        return self.get_key_value(self.current_solution)

    def get_key_value(self, solution: BaseSolution=None) -> float:
        """Get the key value of the solution."""
        pass

    def reset(self, experiment_name: str=None):
        self.current_solution = self.init_solution()
        self.problem_state = self.get_problem_state()
        self.algorithm_data = {}
        self.recording = []
        if experiment_name:
            if os.sep in experiment_name:
                self.output_dir = experiment_name
            else:
                base_output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..", "output") if os.getenv("AMLT_OUTPUT_DIR") else "output"
                self.output_dir = os.path.join(base_output_dir, self.problem, "result", self.data_ref_name, experiment_name)
            os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self, data_path: str) -> dict:
        pass

    def init_solution(self) -> None:
        pass

    def get_problem_state(self, solution: BaseSolution=None) -> dict:
        if solution is None:
            solution = self.current_solution
        problem_state = {
            **self.instance_data,
            "current_solution": solution,
            self.key_item: self.key_value,
            **self.instance_problem_state_function(self.instance_data),
            **self.solution_problem_state_function(self.instance_data, solution, self.get_key_value),
            "get_problem_state": self.get_problem_state,
            "validation_solution": self.validation_solution
        }
        return problem_state

    def validation_solution(self, solution: BaseSolution=None) -> bool:
        """Check the validation of this solution"""
        pass

    def run_heuristic(self, heuristic: callable, parameters:dict={}, inplace: bool=True) -> BaseOperator:
        try:
            operator, delta = heuristic(
                problem_state=self.problem_state,
                algorithm_data=self.algorithm_data,
                **parameters
            )
            if operator is not None:
                result = self.run_operator(operator, inplace, heuristic.__name__)
                if result:
                    self.algorithm_data.update(delta)
                    return operator
            return None
        except Exception as e:
            trace_string = traceback.format_exc()
            print(trace_string)
            return trace_string

    def run_operator(self, operator: BaseOperator, inplace: bool=True, heuristic_name: str=None) -> bool:
        if isinstance(operator, BaseOperator):
            solution = operator.run(self.current_solution)
            if inplace:
                self.current_solution = solution
                self.recording.append((str(heuristic_name), operator))
            self.problem_state = self.get_problem_state()
            return operator
        return None

    def get_observation(self) -> dict:
        pass

    def summarize_env(self) -> str:
        pass

    def dump_result(self, content_dict: dict={}, dump_trajectory: bool=True, dump_heuristic: bool=True, result_file: str="result.txt") -> str:
        content = f"-data: {self.data_path}\n"
        content += f"-current_solution:\n{self.current_solution}\n"
        content += f"-is_complete_solution: {self.is_complete_solution}\n"
        content += f"-is_valid_solution: {self.is_valid_solution}\n"
        content += f"-{self.key_item}: {self.key_value}\n"
        for item, value in content_dict.items():
            content += f"-{item}: {value}\n"
        if dump_trajectory:
            if dump_heuristic:
                trajectory_str = "\n".join([
                    str(index) + "\t" + heuristic_name + "\t" + str(operator)
                    for index, (heuristic_name, operator) in enumerate(self.recording)
                ])
                content += f"-trajectory:\noperation_id\theuristic\toperator(parameter)\n{trajectory_str}\n"
            else:
                trajectory_str = "\n".join([
                    str(index) + "\t" + str(operator)
                    for index, (heuristic_name, operator) in enumerate(self.recording)
                ])
                content += f"-trajectory:\noperation_id\toperator(parameter)\n{trajectory_str}\n"


        if self.output_dir != None and result_file != None:
            output_file = os.path.join(self.output_dir, result_file)
            with open(output_file, "w") as file:  
                file.write(content) 
        
        return content
