import os
import inspect
import importlib
from copy import deepcopy
from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic
from src.problems.base.components import BaseOperator
from src.util.util import load_heuristic


class HeuristicSelectionDataCollector:
    def __init__(
        self,
        problem: str,
        data_name: str,
        score_calculation: callable,
        heuristic_type: str,
        heuristic_pool: list[str],
        search_time: int=1000,
        output_dir: str=None
    ) -> None:
        self.problem = problem
        self.data_name = data_name
        self.heuristic_pool = heuristic_pool
        self.score_calculation = score_calculation
        self.search_time = search_time
        self.output_dir = output_dir

        module = importlib.import_module(f"src.problems.{problem}.env")
        globals()["Env"] = getattr(module, "Env")

        deterministic_heuristics_names, random_heuristics_names = self.filter_deterministic_heuristics(problem, heuristic_pool, data_name)
        heuristic_name_str = ",".join(heuristic_pool)
        deterministic_heuristic_name_str = ",".join(deterministic_heuristics_names)
        random_heuristic_name_str = ",".join(random_heuristics_names)

        os.makedirs(self.output_dir, exist_ok=True)

        information_file = open(os.path.join(output_dir, "information.txt"), "w")
        information_file.write(f"problem: {problem}\n")
        information_file.write(f"data_path: {data_name}\n")
        information_file.write(f"score_calculation: \n{inspect.getsource(score_calculation)}\n")
        information_file.write(f"heuristic type: {heuristic_type}\n")
        information_file.write(f"heuristics: {heuristic_name_str}\n")
        information_file.write(f"deterministic_heuristics: {deterministic_heuristic_name_str}\n")
        information_file.write(f"random_heuristics: {random_heuristic_name_str}\n")
        information_file.write(f"search_time: {search_time}\n")
        information_file.close()
    
    def filter_deterministic_heuristics(self, problem: str, heuristic_pool: list[str], data_name: str) -> list[str]:
        env = Env(data_name)
        previous_steps = int(env.construction_steps / 5)
        try_steps = int(env.construction_steps / 5)
        try_times = 10

        deterministic_heuristics_names = []
        random_heuristics_names = []
        env.reset()
        RandomHyperHeuristic(problem=problem, heuristic_pool=heuristic_pool).run(env, previous_steps)
        saved_env = deepcopy(env)

        for heuristic_file in heuristic_pool:
            heuristic = load_heuristic(heuristic_file, problem=problem)
            key_value = None
            deterministic_flag = True
            for _ in range(try_times):
                env = deepcopy(saved_env)
                for _ in range(try_steps):
                    env.run_heuristic(heuristic)
                if key_value == None:
                    key_value = env.key_value
                if key_value != env.key_value:
                    deterministic_flag = False
                    break

            if deterministic_flag:
                deterministic_heuristics_names.append(heuristic_file.split(".")[0])
            else:
                random_heuristics_names.append(heuristic_file.split(".")[0])
        return deterministic_heuristics_names, random_heuristics_names

    def collect(self):
        env = Env(self.data_name)
        env.reset()
        env.output_dir = self.output_dir
        for round_index in range(int(env.construction_steps * 3)):
            pass
        env.dump_result(True, "finished.txt")