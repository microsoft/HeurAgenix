import os
import inspect
import importlib
import random
import multiprocessing
import multiprocessing.managers
from copy import deepcopy
from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic
from src.util.util import load_heuristic
from src.util.compare_heuristics import compare_heuristics


class HeuristicSelectionDataCollector:
    def __init__(
        self,
        problem: str,
        data_name: str,
        score_calculation: callable,
        heuristic_type: str,
        heuristic_pool: list[str],
        search_time: int=1000,
        save_best: bool=False,
        collection_mode: str="best",
        output_dir: str=None,
    ) -> None:
        self.problem = problem
        self.data_name = data_name
        self.heuristic_pool = heuristic_pool
        self.score_calculation = score_calculation
        self.search_time = search_time
        self.save_best = save_best
        self.collection_mode = collection_mode
        self.output_dir = output_dir

        module = importlib.import_module(f"src.problems.{problem}.env")
        globals()["Env"] = getattr(module, "Env")

        self.deterministic_heuristics_names, self.random_heuristics_names = self.filter_deterministic_heuristics(problem, heuristic_pool, data_name)
        heuristic_name_str = ",".join(heuristic_pool)
        deterministic_heuristic_name_str = ",".join(self.deterministic_heuristics_names)
        random_heuristic_name_str = ",".join(self.random_heuristics_names)

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
        information_file.write(f"collection_mode: {collection_mode}\n")
        
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

    def collect(self, search_interval: int=1, search_time: int=100, max_steps: int=None):
        env = Env(self.data_name)
        env.reset()
        env.output_dir = self.output_dir

        max_steps = max_steps if max_steps else int(env.construction_steps * 3)
        manager = multiprocessing.Manager()
        best_result_proxy = manager.Value('d', float('-inf'))
        records = []
        round_index = 0
        while env.continue_run and round_index <= max_steps:
            output_file_path = os.path.join(self.output_dir, f"round_{round_index}.txt")
            print(f"Search round {round_index} in {output_file_path}")
            # Dump necessary previous information
            output_file = open(output_file_path, "w")
            record_str = "\n".join([f"{record[0]}\t{record[1]}" for record in records])
            output_file.write(f"selected_previous_heuristics\toperators\n{record_str}\n")
            output_file.write("---------------\n")
            output_file.write(f"previous_solution: \n{env.current_solution}\n")
            output_file.write(f"is_complete_solution\t{env.is_complete_solution}\n")
            if env.is_complete_solution:
                output_file.write(f"key_value\t{env.key_value}\n")
            output_file.write("---------------\n")
            
            # Evaluate the performance of each heuristics
            total_results = compare_heuristics(
                env,
                self.heuristic_pool,
                self.heuristic_pool,
                max_steps,
                search_interval,
                search_time,
                self.problem,
                best_result_proxy,
                self.save_best
            )

            # Record and best best for next search
            best_heuristic_name = None
            best_score = None
            best_after_heuristic_env = None
            performances = []
            for heuristic, results, after_step_env, operators in total_results:
                score = None if len(results) <= 0 else self.score_calculation(results)
                results = sorted(results) if env.compare(1, 2) > 0 else sorted(results, reverse=True)
                performances.append([heuristic, str(round(score, 2)), ",".join([str(round(result, 2)) for result in results])])
                if score is not None and best_score is None or env.compare(score, best_score) > 0:
                    best_heuristic_name = heuristic
                    best_operator = str(operators[0])
                    best_score = score
                    best_after_heuristic_env = after_step_env
            if self.collection_mode == "random":
                best_heuristic_name, _, best_after_heuristic_env, operators = random.choice(total_results)
                best_operator = str(operators[0])
            records.append([best_heuristic_name, best_operator])
            output_file.write(f"heuristic\t{self.score_calculation.__name__}\tresults\n")
            output_file.write("\n".join(["\t".join([item for item in performance]) for performance in performances]) + "\n")
            output_file.write("---------------\n")
            output_file.write(f"selected_heuristics\t{best_heuristic_name}\n")
            output_file.write(f"running_operator\t{best_operator}\n")
            output_file.write(f"after_solution: \n{env.current_solution}\n")
            output_file.write(f"is_complete_solution\t{env.is_complete_solution}\n")
            if env.is_complete_solution:
                output_file.write(f"key_value\t{env.key_value}\n")
            output_file.write("---------------\n")
            output_file.close()

            if env.is_complete_solution and best_after_heuristic_env.is_complete_solution and env.compare(env.key_value, best_score) >= 0:
                print(f"Stop as round {round_index}")
                break

            env = best_after_heuristic_env
            round_index += 1
        env.dump_result(dump_trajectory=True, compress_trajectory=False, result_file="finished.txt")