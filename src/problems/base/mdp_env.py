from src.problems.base.env import BaseEnv
from src.problems.base.mdp_components import Solution, ActionOperator

class MDPEnv(BaseEnv):
    """Multi-agents env that stores the static global data, current solution, dynamic state and provide necessary support to the algorithm."""
    def __init__(self, data_name: str, env_class: type, problem: str, **kwargs):
        super().__init__(data_name, problem)
        self.gym_env = env_class(self.data_path)
        self.done = False
        self.reward = 0


    @property
    def is_complete_solution(self) -> bool:
        return self.done

    @property
    def continue_run(self) -> bool:
        return not self.done

    def reset(self, experiment_name: str=None):
        self.gym_env.reset()
        self.done = False
        self.reward = 0
        super().reset(experiment_name)

    def load_data(self, data_path: str) -> None:
        pass

    def init_solution(self) -> None:
        return Solution()

    def get_global_data(self) -> dict:
        global_data = self.gym_env.get_global_data()
        return global_data

    def get_state_data(self) -> dict:
        state_data = self.gym_env.get_state_data()
        state_data["total_reward"] = self.gym_env.get_reward()
        return state_data

    def validation_solution(self, solution: Solution=None) -> bool:
        # TODO
        pass

    def run_operator(self, operator: ActionOperator, inplace: bool=True, heuristic_name: str=None) -> bool:
        if isinstance(operator, ActionOperator):
            solution = operator.run(self.current_solution)
            if inplace:
                self.current_solution = solution
                self.recording.append((str(heuristic_name), operator, str(solution)))
            _, reward, self.done, _ = self.gym_env.step(operator.actions)
            self.reward += reward
            self.state_data = self.get_state_data()
            return True
        return False

    def dump_result(self, dump_trajectory: bool=True, compress_trajectory: bool=False, result_file: str="result.txt") -> str:
        content_dict = self.get_state_data()
        content = super().dump_result(content_dict=content_dict, dump_trajectory=dump_trajectory, compress_trajectory=compress_trajectory, result_file=result_file)
        return content
    
    def summarize_env(self) -> str:
        if hasattr(self.gym_env, "summarize_env"):
            return self.gym_env.summarize_env()
        return None