import os
from src.pipeline.heuristic_generator import HeuristicGenerator
from src.util.gpt_helper import GPTHelper

problem = "tsp"
paper_path = "fast_minimum_weight_double_tree_shortcutting_for_metric_TSP.tex"
smoke_test = True
gpt_helper = GPTHelper(
    prompt_dir=os.path.join("src", "problems", "base", "prompt"),
    output_dir=os.path.join("output", problem, "generate_heuristic")
)
heuristic_generator = HeuristicGenerator(gpt_helper=gpt_helper, problem=problem)
heuristic_generator.generate_from_paper(paper_path, smoke_test=smoke_test)