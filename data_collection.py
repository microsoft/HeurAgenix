import os
import importlib
import copy
import numpy as np
from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic
from src.util.util import load_heuristic

problem = "tsp"
data_name = "bayg29.tsp"
try_time = 1000
top_k = 20
r = 1

module = importlib.import_module(f"src.problems.{problem}.env")
globals()["Env"] = getattr(module, "Env")
env = Env(data_name, mode="test")

heuristic_dir = os.path.join("src", "problems", problem, "heuristics", "basic_heuristics")
random_hh = RandomHyperHeuristic(heuristic_dir, problem=problem)
output_file = open(f"{data_name}.result", "w")
output_file.write(data_name + "\n")
output_file.write(f"try time: {try_time}\n")
output_file.write(f"top_k: {top_k}\n")
output_file.write(f"r: {r}\n")

selected_previous_heuristics = []
for round in range(int(env.construction_steps * 2)):
    best_score = np.inf
    for heuristic in os.listdir(heuristic_dir):
        env.reset(f"round_{round}_{heuristic}")
        for h in selected_previous_heuristics:
            env.run_heuristic(load_heuristic(h, heuristic_dir))
        env.run_heuristic(load_heuristic(heuristic, heuristic_dir))
        saved_solution = copy.deepcopy(env.current_solution)
        saved_state = copy.deepcopy(env.state_data)
        saved_algorithm_data = copy.deepcopy(env.algorithm_data)

        results = []
        for _ in range(try_time):
            env.current_solution = saved_solution
            env.state_data = saved_state
            env.algorithm_data = saved_algorithm_data
            random_hh.run(env)
            results.append(env.key_value)
            if (_ + 1) % 200 == 0:
                results = sorted(results)
                average = sum(results) / len(results)
                top_k_average = sum(results[:top_k]) / top_k
                score = r * average + (1 - r) * top_k_average
                if score > best_score * 1.01:
                    break
        results = sorted(results)
        average = sum(results) / len(results)
        top_k_average = sum(results[:top_k]) / top_k
        score = r * average + (1 - r) * top_k_average
        # r *= 1.01
        output_file.write(heuristic + ", " + str(_) + ", " + str(average) + ", " + str(top_k_average) + ", " + str(score) + ", " + str(results) + "\n")
        if score < best_score:
            best_score = score
            best_heuristics = heuristic
    selected_previous_heuristics.append(best_heuristics)
    output_file.write(best_heuristics + "\n---------------\n")
output_file.close()