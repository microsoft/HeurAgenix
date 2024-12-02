import os
from src.problems.road_charging.env import Env
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic


heuristic = "random_dc6e"
print(f"Running heuristic algorithm {heuristic}...")

env = Env(data_name="case_1.json")
env.reset("debug")
env.summarize_env()

hyper_heuristic = SingleHyperHeuristic(heuristic, problem="road_charging")
hyper_heuristic.run(env)
env.dump_result()
print(os.path.join(env.output_dir, "result.txt"), env.key_item, env.key_value)
