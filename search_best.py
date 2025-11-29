import os
import sys
from datetime import datetime
from src.problems.max_cut.env import Env
from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic

iterations_scale_factor = 5
def run_once(data_name: str, heuristic_dir: str) -> float:
    env = Env(data_name=data_name)
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..", "output") if os.getenv("AMLT_OUTPUT_DIR") else "output"
    output_dir = os.path.join(base_output_dir, "max_cut", "search_best_result", env.data_ref_name, datetime_str)
    env.reset(output_dir=output_dir)
    algorithm = RandomHyperHeuristic(os.listdir(heuristic_dir), "max_cut", iterations_scale_factor)
    result = algorithm.run(env)
    return result

def main():
    data_name = sys.argv[1]
    heuristic_dir = "evolved_heuristics.part2"
    if len(sys.argv) > 2:
        heuristic_dir = sys.argv[2]

    heuristic_dir = os.path.join("src", "problems", "max_cut", "heuristics", heuristic_dir)
    result = run_once(data_name, heuristic_dir)
    print(result)

main()