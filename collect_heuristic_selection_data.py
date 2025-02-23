import argparse
import os
from datetime import datetime
from src.pipeline.heuristic_selection_data_collector import HeuristicSelectionDataCollector
from src.util.util import load_heuristic


def parse_arguments():
    problem_pool = [problem for problem in os.listdir(os.path.join("src", "problems")) if problem != "base"]

    parser = argparse.ArgumentParser(description="Generate heuristic")
    parser.add_argument("-p", "--problem", choices=problem_pool, required=True, help="Type of problem to solve.")
    parser.add_argument("-c", "--collection_case", type=str, required=True, help="Data name for collection case")
    parser.add_argument("-d", "--heuristic_type", type=str, default="basic", choices=["basic", "evolved"], help="Directory containing heuristic functions.")
    parser.add_argument("-s", "--search_time", default=1000, type=int, help="Search times for each heuristic.")    
    parser.add_argument("-r", "--score_calculation", choices=["average_score", "a8t2"], default="average_score", help="Function to calculate score.")
    parser.add_argument("-fd", "--folder_dir", default=None, help="Path of result folder dir")

    return parser.parse_args()

def a8t2(results: list[float]) -> float:
    top_k = 20
    average_score_ratio = 0.8
    average_score = sum(results) / len(results)
    top_k_score = sum(sorted(results[: top_k])) / top_k
    score = average_score_ratio * average_score + (1 - average_score_ratio) * top_k_score
    return score

def average_score(results: list[float]) -> float:
    average_score = sum(results) / len(results)
    return average_score

def main():
    args = parse_arguments()
    problem = args.problem
    data_name = args.collection_case
    heuristic_type = args.heuristic_type
    search_time = args.search_time
    folder_dir = args.folder_dir
    score_calculation = eval(args.score_calculation)

    if heuristic_type == "basic":
        heuristic_pool = os.listdir(os.path.join("src", "problems", problem, "heuristics", "basic_heuristics"))
    elif heuristic_type == "evolved":
        heuristic_pool = os.listdir(os.path.join("src", "problems", problem, "heuristics", "evolved_heuristics"))
        evolved_names = [heuristic[:-8] for heuristic in heuristic_pool]
        heuristic_pool += [file for file in os.listdir(os.path.join("src", "problems", problem, "heuristics", "basic_heuristics")) if file[:-8] not in evolved_names]

    base_output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..", "output") if os.getenv("AMLT_OUTPUT_DIR") else "output"
    os.makedirs(base_output_dir, exist_ok=True)
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if folder_dir:
        output_dir = os.path.join(base_output_dir, problem, "heuristic_selection_data_collection", folder_dir, f"{data_name}.{heuristic_type}.{datetime_str}.result")
    else:
        output_dir = os.path.join(base_output_dir, problem, "heuristic_selection_data_collection", f"{data_name}.{heuristic_type}.{datetime_str}.result")
    print(f"Collect data in {output_dir}")
    data_collector = HeuristicSelectionDataCollector(problem, data_name, score_calculation, heuristic_type, heuristic_pool, search_time, output_dir)
    data_collector.collect(1, search_time)
    print(f"Done")

if __name__ == "__main__":
    main()    
