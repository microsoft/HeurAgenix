import os
from src.pipeline.heuristic_evolver import HeuristicEvolver
from src.util.gpt_helper import GPTHelper

problem = "tsp"
basic_heuristic = "nearest_neighbor_f91d"
perturbation_ratio = 0.1
perturbation_time = 1000
filtered_num = 3
evolution_round = 3
time_limitation = 10
smoke_test = True
validation = True

perturbation_heuristic = {
    "tsp": "random_80a0",
    "cvrp": "random_bfdc",
    "jssp": "random_6512",
    "max_cut": "random_5c59",
    "mkp": "random_4c25",
    "dposp": "random_c05a"
}[problem]

heuristic_dir = os.path.join("src", "problems", problem, "heuristics", "basic_heuristics")
train_dir = os.path.join("src", "problems", problem, "data", "train_data")
validation_dir = os.path.join("src", "problems", problem, "data", "validation_data")

gpt_helper = GPTHelper(prompt_dir=os.path.join("src", "problems", "base", "prompt"))
gpt_evolver = HeuristicEvolver(gpt_helper, problem, heuristic_dir, train_dir, validation_dir)
evolved_heuristics = gpt_evolver.evolution(
    basic_heuristic,
    perturbation_heuristic,
    perturbation_ratio=0.1,
    perturbation_time=perturbation_time,
    filtered_num=filtered_num,
    evolution_round=evolution_round,
    time_limitation=time_limitation,
    smoke_test=smoke_test,
    validation=validation
)
print(evolved_heuristics)