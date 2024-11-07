import os
from src.pipeline.heuristic_generator import HeuristicGenerator
from src.util.gpt_helper import GPTHelper

problem = "tsp"
smoke_test = True
gpt_helper = GPTHelper(
    prompt_dir=os.path.join("src", "problems", "base", "prompt"),
    output_dir=os.path.join("output", problem, "generate_heuristic")
)
heuristic_generator = HeuristicGenerator(gpt_helper=gpt_helper, problem=problem)
heuristic_generator.generate_from_gpt(smoke_test=smoke_test)