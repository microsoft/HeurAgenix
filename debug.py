import os
from src.problems.road_charging.env import Env
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic


heuristic = "random_dc6e"
print(f"Running heuristic algorithm {heuristic}...")

env = Env(data_name="src/problems/road_charging/data/config.json")
env.reset("debug")

hyper_heuristic = SingleHyperHeuristic(heuristic, problem="road_charging")
hyper_heuristic.run(env)

print(os.path.join(env.output_dir, "result.txt"), env.key_item, env.key_value)
