from copy import deepcopy
import os
import json
import importlib
import re
import shutil
import pandas as pd
import traceback
from io import StringIO
from src.problems.base.env import BaseEnv
from src.pipeline.heuristic_generator import HeuristicGenerator
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic
from src.pipeline.hyper_heuristics.perturbation import PerturbationHyperHeuristic
from src.util.util import extract, filter_dict_to_str, parse_text_to_dict, load_heuristic, extract_function_with_short_docstring, search_file
from src.util.gpt_helper import GPTHelper

class HeuristicEvolver:
    def __init__(
        self,
        gpt_helper: GPTHelper,
        problem: str,
        train_dir: str=None,
        validation_dir: str=None,
    ) -> None:
        self.gpt_helper = gpt_helper
        self.problem = problem
        self.train_dir = train_dir if train_dir is not None else os.path.join("src", "problems", problem, "data", "train_data")
        self.validation_dir = validation_dir if validation_dir is not None else os.path.join("src", "problems", problem, "data", "validation_data")
        module = importlib.import_module(f"src.problems.{problem}.env")
        globals()["Env"] = getattr(module, "Env")
    
    def evolution(
            self,
            basic_heuristic_file: str,
            perturbation_heuristic_file: str,
            perturbation_ratio: float=0.1,
            perturbation_time: int=100,
            filtered_num: int=3,
            evolution_round: int=3,
            time_limitation: float=10,
            smoke_test: bool=True,
            validation: bool=True,
        ) -> None:

        # Prepare other heuristics' description for this evolution
        heuristic_dir = os.path.dirname(basic_heuristic_file)
        all_heuristic_docs = "\n".join([
            extract_function_with_short_docstring(open(os.path.join(heuristic_dir, heuristic_file)).read(), heuristic_file.split(".")[0])
            for heuristic_file in os.listdir(heuristic_dir)
        ])
        # Getting baseline for basic heuristic
        print(f"Getting baselines for {basic_heuristic_file}")
        baselines = self.validation(self.validation_dir, basic_heuristic_file, validation=validation)

        total_heuristic_files = [(basic_heuristic_file, 0)]
        for _ in range(evolution_round):
            # Filter the best heuristics
            filtered_heuristic_files = sorted(total_heuristic_files, key=lambda x: x[1], reverse=True)[: filtered_num]
            for basic_heuristic_file, _ in filtered_heuristic_files:
                basic_heuristic = basic_heuristic_file.split(os.sep)[-1].split('.')[0]
                for data_name in os.listdir(self.train_dir):
                    env = Env(data_name=data_name)
                    # Perturbation
                    print(f"Perturb {basic_heuristic} on {data_name}")
                    negative_result_file, positive_result_file = self.perturbation(
                        env,
                        basic_heuristic_file,
                        perturbation_heuristic_file,
                        perturbation_ratio,
                        perturbation_time
                    )

                    # Evolution
                    evolved_heuristic_files = []
                    if positive_result_file:
                        print(f"Evolution {basic_heuristic} on {data_name}")
                        try:
                            evolved_heuristic_files = self.comparison(
                                env,
                                positive_result_file,
                                negative_result_file,
                                basic_heuristic_file,
                                all_heuristic_docs,
                                os.path.join("output", self.problem, "train_result", data_name, f"{basic_heuristic}.evolution"),
                                smoke_test
                            )
                        except Exception as e:
                            trace_string = traceback.format_exc()
                            self.gpt_helper.load(trace_string)
                            self.gpt_helper.dump("Error")
                            self.gpt_helper.messages.pop()
                            continue
                
                    # validation
                    if len(evolved_heuristic_files) > 0:
                        for evolved_heuristic_file in evolved_heuristic_files:
                            print(f"Validation on {evolved_heuristic_file}")
                            validation_values = self.validation(self.validation_dir, evolved_heuristic_file, time_limitation, True)
                            if validation_values:
                                average_advantage = 0
                                for index in range(len(validation_values)):
                                    average_advantage += env.compare(validation_values[index], baselines[index]) / baselines[index]
                                average_advantage /= len(validation_values)
                                print(f"Evolved heuristic:{evolved_heuristic_file}; average_advantage: {average_advantage}")
                                total_heuristic_files.append((evolved_heuristic_file, average_advantage))
                            else:
                                print(f"Abandon {evolved_heuristic_file}")
        return filtered_heuristic_files
 
    def perturbation(
            self,
            env: BaseEnv,
            basic_heuristic_file: str,
            perturbation_heuristic_file: str,
            perturbation_ratio: float=0.1,
            perturbation_time: int=100
        ) -> tuple[bool, str, str]:
        basic_heuristic = basic_heuristic_file.split(os.sep)[-1].split('.')[0]
        env.reset(f"{basic_heuristic}.negative")

        # Generate negative result from basic heuristic
        hyper_heuristic = SingleHyperHeuristic(basic_heuristic_file, problem=self.problem)
        hyper_heuristic.run(env)
        env.dump_result(dump_trajectory=True)
        negative_result_file = os.path.join(env.output_dir, "result.txt")
        negative_value = env.key_value

        # Generate positive result by perturbation heuristic
        positive_result_file = None
        for _ in range(perturbation_time):
            env.reset(f"{basic_heuristic}.positive")
            hyper_heuristic = PerturbationHyperHeuristic(basic_heuristic_file, perturbation_heuristic_file, perturbation_ratio, problem=self.problem)
            hyper_heuristic.run(env)
            if env.compare(env.key_value, negative_value) > 0:
                env.dump_result(dump_trajectory=True)
                positive_result_file = os.path.join(env.output_dir, "result.txt")
                break
        return negative_result_file, positive_result_file

    def comparison(
            self,
            env: BaseEnv,
            positive_result_file: str,
            negative_result_file: str,
            heuristic_file: str,
            all_heuristic_docs: str,
            output_dir: str,
            smoke_test: bool=True,
        ) -> list[str]:
        env.reset()
        self.gpt_helper.reset(output_dir)
        output_heuristic_files = []
        
        shutil.copyfile(positive_result_file, os.path.join(output_dir, "positive_solution.txt"))
        shutil.copyfile(negative_result_file, os.path.join(output_dir, "negative_solution.txt"))
        if os.path.isdir(env.data_path):
            shutil.copytree(env.data_path, os.path.join(output_dir, os.path.basename(env.data_path)), dirs_exist_ok=True)
        else:
            shutil.copyfile(env.data_path, os.path.join(output_dir, os.path.basename(env.data_path)))

        # Load background
        prompt_dict = self.gpt_helper.load_background(self.problem)
        prompt_dict["all_heuristic_docs"] = all_heuristic_docs

        # Load components
        if os.path.exists(os.path.join("src", "problems", self.problem, "components.py")):
            module = importlib.import_module(f"src.problems.{self.problem}.components")
        else:
            module = importlib.import_module(f"src.problems.base.mdp_components")
        names_to_import = (name for name in dir(module) if not name.startswith('_'))
        for name in names_to_import:
            globals()[name] = getattr(module, name)

        # Load data feature
        get_global_data_feature_function = load_heuristic("evaluation_function.py", problem=self.problem, function_name="get_global_data_feature")
        get_state_data_feature_function = load_heuristic("evaluation_function.py", problem=self.problem, function_name="get_state_data_feature")
        prompt_dict["global_data"] = filter_dict_to_str(env.global_data)
        prompt_dict["global_data_feature"] = filter_dict_to_str(get_global_data_feature_function(env.global_data))

        # Load solution
        positive_result = parse_text_to_dict(open(positive_result_file).read())
        negative_result = parse_text_to_dict(open(negative_result_file).read())
        prompt_dict["positive_solution"] = positive_result["current_solution"]
        prompt_dict["negative_solution"] = negative_result["current_solution"]
        prompt_dict["positive_result"] = positive_result[env.key_item]
        prompt_dict["negative_result"] = negative_result[env.key_item]
        prompt_dict["positive_trajectory"] = positive_result["trajectory"]
        prompt_dict["negative_trajectory"] = negative_result["trajectory"]
        negative_trajectory_df = pd.read_csv(StringIO(negative_result["trajectory"]), sep="\t")

        # Load heuristic code
        heuristic_file = search_file(heuristic_file, problem=self.problem)
        function_name = heuristic_file.split(os.sep)[-1].split(".")[0]
        function_code = open(heuristic_file).read()
        heuristic_name = function_name[:-5]
        prompt_dict["function_name"] = function_name
        prompt_dict["function_code"] = function_code
        prompt_dict["heuristic_name"] = heuristic_name

        # Identify bottleneck operations
        self.gpt_helper.load("identify_bottleneck_v2", prompt_dict)
        response = self.gpt_helper.chat()
        bottleneck_operations = extract(response, key="bottleneck_operations", sep="\n")
        self.gpt_helper.dump("bottleneck_operations")

        for index, bottleneck_operation_analysis in enumerate(bottleneck_operations):
            self.gpt_helper.load_chat("bottleneck_operations")
            # Reproduce the state before bottleneck
            bottleneck_operation_id, proposed_operation, reason = bottleneck_operation_analysis.split(";")
            bottleneck_operation_id = int(re.search(r'\d+', bottleneck_operation_id).group())
            bottleneck_operation = list(negative_trajectory_df[negative_trajectory_df["operation_id"] == bottleneck_operation_id]["operator(parameter)"])[0]
            prompt_dict["bottleneck_operation_id"] = bottleneck_operation_id
            prompt_dict["bottleneck_operation"] = bottleneck_operation
            prompt_dict["proposed_operation"] = proposed_operation
            env.reset()
            for previous_operation in negative_trajectory_df[negative_trajectory_df["operation_id"] < bottleneck_operation_id]["operator(parameter)"]:
                env.run_operator(eval(previous_operation))
            prompt_dict["state_data"] = filter_dict_to_str(env.state_data)
            prompt_dict["state_data_feature"] = filter_dict_to_str(get_state_data_feature_function(env.global_data, env.state_data))

            # Try to provide suggestion
            self.gpt_helper.load("extract_suggestion_v2", prompt_dict)
            response = self.gpt_helper.chat()
            suggestion = extract(response, key="suggestion")
            if suggestion is None:
                continue
            self.gpt_helper.dump(f"suggestion_{index}")
            # Implement the new code
            description = f"Now, based on these suggestions:\n{suggestion}\nUpdate the nearest_neighbor_f91d."
            output_heuristic_file = HeuristicGenerator(self.gpt_helper, self.problem).generate(heuristic_name, description, smoke_test)
            output_heuristic_files.append(output_heuristic_file)
        return output_heuristic_files

    def validation(self, validation_dir: str, heuristic_file: str, time_limitation: float=10, validation: bool=True) -> list[float, float]:
        validation_results = []
        heuristic_name = heuristic_file.split(os.sep)[-1].split(".py")[0]
        for data_name in os.listdir(validation_dir):
            env = Env(data_name=os.path.join(validation_dir, data_name))
            env.reset(heuristic_name)
            hyper_heuristic = SingleHyperHeuristic(heuristic_file, problem=self.problem)
            is_complete_solution = hyper_heuristic.run(env, time_limitation, validation=validation)
            if is_complete_solution:
                env.dump_result(dump_trajectory=False)
                validation_results.append(env.key_value)
            else:
                return None
        return validation_results