import os
from src.pipeline.evaluation_function_generator import EvaluationFunctionGenerator
from src.util.gpt_helper import GPTHelper

problem = "tsp"
smoke_test = True
gpt_helper = GPTHelper(
    prompt_dir=os.path.join("src", "problems", "base", "prompt"),
    output_dir=os.path.join("output", problem, "generate_evaluation_function")
)
evaluation_function_generator = EvaluationFunctionGenerator(gpt_helper=gpt_helper, problem=problem)
evaluation_function_generator.generate_evaluation_function(smoke_test=smoke_test)