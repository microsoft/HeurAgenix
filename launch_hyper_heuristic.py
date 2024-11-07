import os
import importlib
from datetime import datetime
from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic
from src.pipeline.hyper_heuristics.single_construct_single_improve import SingleConstructiveSingleImproveHyperHeuristic
from src.pipeline.hyper_heuristics.gpt_selection import GPTSelectionHyperHeuristic
from src.util.gpt_helper import GPTHelper

problem = "tsp"
heuristic_dir = "basic_heuristics"
validation_for_each_step = True
hhs = ["nearest_neighbor_f91d", "cheapest_insertion_605f", "farthest_insertion_b6d3", "random_hh", "gpt_hh"] # tsp
# hhs = ["nearest_neighbor_99ba", "min_cost_insertion_7bfa", "farthest_insertion_ce2b", "random_hh", "gpt_hh"] # cvrp
# hhs = ["most_work_remaining_930e", "first_come_first_served_6c4f", "shortest_processing_time_first_c374", "random_hh", "gpt_hh"] # jssp
# hhs = ["most_weight_neighbors_320c", "highest_weight_edge_eb0c", "balanced_cut_21d5", "random_hh", "gpt_hh"] # max_cut
# hhs = ["greedy_by_profit_8df3", "greedy_by_weight_ece2", "greedy_by_density_9e8d", "random_hh", "gpt_hh"] # mkp
# hhs = ["shortest_operation_ff40", "least_order_remaining_9c3c", "greedy_by_order_density_c702" "random_hh", "gpt_hh", "or_solver"] # dposp

module = importlib.import_module(f"src.problems.{problem}.env")
globals()["Env"] = getattr(module, "Env")

problem_dir = os.path.join("src", "problems", problem)
test_data_dir = os.path.join(problem_dir, "data", "test_data")
heuristic_dir = os.path.join(problem_dir, "heuristics", heuristic_dir)
test_data_names = os.listdir(test_data_dir)
all_heuristics = [file.split(".")[0] for file in os.listdir(heuristic_dir)]

for data_name in test_data_names:
    print(data_name)
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    env = Env(data_name=data_name, mode="test")
    for hh in hhs:
        hh = hh.split(".")[0]
        if hh in all_heuristics:
            env.reset(hh)
            hyper_heuristic = SingleHyperHeuristic(hh, problem=problem, heuristic_dir=heuristic_dir)
        elif hh == "gpt_hh":
            env.reset(f"gpt_hh.{datetime_str}")
            gpt_helper = GPTHelper(
                prompt_dir=os.path.join("src", "problems", "base", "prompt"),
                output_dir=env.output_dir,
            )
            hyper_heuristic = GPTSelectionHyperHeuristic(gpt_helper=gpt_helper, problem=problem, heuristic_dir=heuristic_dir)
        elif hh == "random_hh":
            env.reset(f"random_hh.{datetime_str}")
            hyper_heuristic = RandomHyperHeuristic(problem=problem, heuristic_dir=heuristic_dir)
        elif hh == "or_solver":
            env.reset(f"or_solver")
            module = importlib.import_module(f"src.problems.{problem}.or_solver")
            globals()["ORSolver"] = getattr(module, "ORSolver")
            hyper_heuristic = ORSolver(problem=problem)

        complete = hyper_heuristic.run(env, validation=validation_for_each_step)
        if complete:
            env.dump_result()
            print(os.path.join(env.output_dir, "result.txt"), env.key_item, env.key_value)
