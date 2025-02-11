import os
import importlib
import re
import pandas as pd
import traceback
from io import StringIO
from src.problems.base.env import BaseEnv
from src.pipeline.heuristic_generator import HeuristicGenerator
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic
from src.pipeline.hyper_heuristics.perturbation import PerturbationHyperHeuristic
from src.util.util import df_to_str, extract, filter_dict_to_str, parse_text_to_dict, load_heuristic, extract_function_with_short_docstring, search_file
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
        train_dir = train_dir if train_dir is not None else os.path.join("src", "problems", problem, "data", "train_data")
        validation_dir = validation_dir if validation_dir is not None else os.path.join("src", "problems", problem, "data", "validation_data")
        self.train_cases = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
        self.validation_cases = [os.path.join(validation_dir, f) for f in os.listdir(validation_dir)]
        self.get_global_data_feature_function = load_heuristic("evaluation_function.py", problem=self.problem, function_name="get_global_data_feature")
        self.get_state_data_feature_function = load_heuristic("evaluation_function.py", problem=self.problem, function_name="get_state_data_feature")

        # Load env
        module = importlib.import_module(f"src.problems.{problem}.env")
        globals()["Env"] = getattr(module, "Env")

        # Load components
        if os.path.exists(os.path.join("src", "problems", self.problem, "components.py")):
            module = importlib.import_module(f"src.problems.{self.problem}.components")
        else:
            module = importlib.import_module(f"src.problems.base.mdp_components")
        names_to_import = (name for name in dir(module) if not name.startswith('_'))
        for name in names_to_import:
            globals()[name] = getattr(module, name)
        
        # Ready for validation feature
        global_features = []
        for data in self.validation_cases:
            global_data = Env(data_name=data).get_global_data()
            global_feature = {"data_name": data.split(os.sep)[-1]}
            global_feature.update(self.get_global_data_feature_function(global_data))
            global_features.append(global_feature)
        self.global_feature_df = pd.DataFrame(global_features)

    def evolution(
            self,
            basic_heuristic_file: str,
            perturbation_heuristic_file: str,
            perturbation_ratio: float=0.1,
            perturbation_time: int=100,
            filtered_num: int=3,
            evolution_round: int=3,
            max_refinement_round: int=5,
            time_limitation: float=10,
            smoke_test: bool=True,
        ) -> None:

        # Prepare other heuristics' description for this evolution
        heuristic_dir = os.path.dirname(basic_heuristic_file)
        all_heuristic_docs = "\n".join([
            open(os.path.join(heuristic_dir, heuristic_file)).read()
            for heuristic_file in os.listdir(heuristic_dir)
        ])

        total_heuristic_benchmarks = [(basic_heuristic_file, 0)]
        for _ in range(evolution_round):
            # Filter the best heuristics
            filtered_heuristic_benchmarks = sorted(total_heuristic_benchmarks, key=lambda x: x[1], reverse=True)[: filtered_num]
            for basic_heuristic_file, _ in filtered_heuristic_benchmarks:
                for data_name in self.train_cases:
                    evolved_heuristic_with_improvements = self.evolution_single(
                        train_data=data_name,
                        basic_heuristic_file=basic_heuristic_file,
                        perturbation_heuristic_file=perturbation_heuristic_file,
                        all_heuristic_docs=all_heuristic_docs,
                        perturbation_ratio=perturbation_ratio,
                        perturbation_time=perturbation_time,
                        max_refinement_round=max_refinement_round,
                        time_limitation=time_limitation,
                        smoke_test=smoke_test
                    )
                    total_heuristic_benchmarks.extend(evolved_heuristic_with_improvements)
        return filtered_heuristic_benchmarks

    def evolution_single(
            self,
            train_data: str,
            basic_heuristic_file: str,
            perturbation_heuristic_file: str,
            all_heuristic_docs: str,
            perturbation_ratio: float=0.1,
            perturbation_time: int=100,
            max_refinement_round: int=5,
            time_limitation: float=10,
            smoke_test: bool=True
    ) -> list[tuple[str, list[float]]]:
        try:
            data_name = train_data.split(os.sep)[-1]
            env = Env(data_name=train_data)
            basic_heuristic_name = basic_heuristic_file.split(os.sep)[-1].split(".")[0]
            output_dir = os.path.join("output", self.problem, "evolution_result", data_name, f"{basic_heuristic_name}.evolution")
            self.gpt_helper.reset(output_dir)

            # Perturb for better solution
            negative_result_file, positive_result_file = self.perturbation(
                env,
                basic_heuristic_file,
                perturbation_heuristic_file,
                output_dir,
                perturbation_ratio,
                perturbation_time,
            )

            refined_heuristic_benchmarks = []
            if positive_result_file:
                print(f"Evolution {basic_heuristic_name} on {data_name}")
                # Get bassline 
                basic_heuristic_result = self.validation(self.validation_cases, basic_heuristic_file)

                prompt_dict = self.gpt_helper.load_background(self.problem)
                prompt_dict["all_heuristic_docs"] = all_heuristic_docs
                self.load_heuristic_code(basic_heuristic_file, prompt_dict)

                # Identity bottlenecks
                bottlenecks = self.identity_bottlenecks(
                    prompt_dict=prompt_dict,
                    env=env,
                    positive_result_file=positive_result_file,
                    negative_result_file=negative_result_file,
                    heuristic_file=basic_heuristic_file
                )

                for bottleneck_index, (bottleneck_operation_id, proposed_operation, reason) in enumerate(bottlenecks):
                    # Raise suggestion and provide evolved heuristics
                    basic_heuristic_result = self.validation(self.validation_cases, basic_heuristic_file)
                    suggestion_name = f"suggestion_{bottleneck_index}"
                    suggested_heuristic_file, suggestion, suggested_result = self.raise_suggestion(
                        prompt_dict=prompt_dict,
                        env=env,
                        bottleneck_operation_id=bottleneck_operation_id,
                        proposed_operation=proposed_operation,
                        reason=reason,
                        suggestion_name=suggestion_name,
                        smoke_test=smoke_test
                    )
                    if suggested_heuristic_file:
                        suggested_improvement = sum(self.get_improvement(env, basic_heuristic_result, suggested_result)) / len(basic_heuristic_result)
                        print(f"Improvement for {suggested_heuristic_file}: {suggested_improvement}")
                        refined_heuristic_benchmarks.append([suggested_heuristic_file, suggested_improvement])
                        # Fine tune the evolved heuristics
                        previous_heuristic_name = basic_heuristic_name
                        previous_heuristic_result = basic_heuristic_result
                        last_heuristic_name = suggested_heuristic_file.split(os.sep)[-1].split(".")[0]
                        last_heuristic_result = suggested_result
                        last_suggestion = suggestion
                        for refine_index in range(max_refinement_round):
                            suggestion_name = f"suggestion_{bottleneck_index}_refine_{refine_index}"
                            refined_heuristic_file, suggestion, refined_result = self.refine_heuristic(
                                prompt_dict=prompt_dict,
                                env=env,
                                basic_heuristic_name=basic_heuristic_name,
                                basic_heuristic_result=basic_heuristic_result,
                                previous_heuristic_name=previous_heuristic_name,
                                previous_heuristic_result=previous_heuristic_result,
                                last_heuristic_name=last_heuristic_name,
                                last_heuristic_result=last_heuristic_result,
                                last_suggestion=last_suggestion,
                                suggestion_name=suggestion_name,
                                time_limitation=time_limitation,
                                smoke_test=smoke_test
                            )
                            if None in refined_result:
                                print("Error and skip")
                                continue
                            if refined_heuristic_file:
                                refined_improvement = sum(self.get_improvement(env, basic_heuristic_result, refined_result)) / len(basic_heuristic_result)
                                print(f"Improvement for {refined_heuristic_file}: {refined_improvement}")
                                refined_heuristic_benchmarks.append([refined_heuristic_file, refined_improvement])
                                if suggestion is None:
                                    break
                                previous_heuristic_name = last_heuristic_name
                                previous_heuristic_result = last_heuristic_result
                                last_suggestion = suggestion
                                last_heuristic_name = refined_heuristic_file.split(os.sep)[-1].split(".")[0]
                                last_heuristic_result = refined_result
        except Exception as e:
            trace_string = traceback.format_exc()
            print(trace_string)

        return refined_heuristic_benchmarks

    def perturbation(
            self,
            env: BaseEnv,
            basic_heuristic_file: str,
            perturbation_heuristic_file: str,
            output_dir: str,
            perturbation_ratio: float=0.1,
            perturbation_time: int=100,
        ) -> tuple[bool, str, str]:
        env.reset(os.path.join(output_dir, "negative_solution"))

        # Generate negative result from basic heuristic
        hyper_heuristic = SingleHyperHeuristic(basic_heuristic_file, problem=self.problem)
        hyper_heuristic.run(env)
        env.dump_result(dump_trajectory=True)
        negative_result_file = os.path.join(env.output_dir, "result.txt")
        negative_value = env.key_value

        # Generate positive result by perturbation heuristic
        positive_result_file = None
        for _ in range(perturbation_time):
            env.reset(os.path.join(output_dir, "positive_solution"))
            hyper_heuristic = PerturbationHyperHeuristic(basic_heuristic_file, perturbation_heuristic_file, self.problem, perturbation_ratio)
            hyper_heuristic.run(env)
            if env.compare(env.key_value, negative_value) > 0:
                env.dump_result(dump_trajectory=True)
                positive_result_file = os.path.join(env.output_dir, "result.txt")
                break
        return negative_result_file, positive_result_file

    def load_heuristic_code(self, heuristic_file: str, prompt_dict: dict) -> str:
        heuristic_file = search_file(heuristic_file, problem=self.problem)
        function_name = heuristic_file.split(os.sep)[-1].split(".")[0]
        function_code = open(heuristic_file).read()
        heuristic_name = function_name[:-5]
        prompt_dict["function_name"] = function_name
        prompt_dict["function_code"] = function_code
        prompt_dict["heuristic_name"] = heuristic_name
        return function_code

    def identity_bottlenecks(
            self,
            prompt_dict: dict,
            env: BaseEnv,
            positive_result_file: str,
            negative_result_file: str,
            heuristic_file: str
        ) -> list[list[int, str, str, str]]:
        env.reset()

        # Load data feature
        prompt_dict["global_data"] = filter_dict_to_str(env.global_data)
        prompt_dict["global_data_feature"] = filter_dict_to_str(self.get_global_data_feature_function(env.global_data))

        # Load solution
        positive_result = parse_text_to_dict(open(positive_result_file).read())
        negative_result = parse_text_to_dict(open(negative_result_file).read())
        prompt_dict["positive_solution"] = positive_result["current_solution"]
        prompt_dict["negative_solution"] = negative_result["current_solution"]
        prompt_dict["positive_result"] = positive_result[env.key_item]
        prompt_dict["negative_result"] = negative_result[env.key_item]
        prompt_dict["positive_trajectory"] = positive_result["trajectory"]
        prompt_dict["negative_trajectory"] = negative_result["trajectory"]

        # Identify bottleneck operations
        self.gpt_helper.load("identify_bottleneck_v2", prompt_dict)
        response = self.gpt_helper.chat()
        bottleneck_operation_strs = extract(response, key="bottleneck_operations", sep="\n")
        self.gpt_helper.dump("bottleneck_operations")

        bottlenecks = []
        for bottleneck_operation_str in bottleneck_operation_strs:
            # Reproduce the state before bottleneck
            bottleneck_operation_id, proposed_operation, reason = bottleneck_operation_str.split(";")
            bottleneck_operation_id = int(re.search(r'\d+', bottleneck_operation_id).group())
            bottlenecks.append([bottleneck_operation_id, proposed_operation, reason])

        return bottlenecks

    def raise_suggestion(
            self,
            prompt_dict: dict,
            env: BaseEnv,
            bottleneck_operation_id: int,
            proposed_operation: str,
            reason: str,
            suggestion_name: str,
            smoke_test: bool,
    ) -> tuple[str, str]:
        env.reset()
        prompt_dict["bottleneck_operation_id"] = bottleneck_operation_id
        prompt_dict["proposed_operation"] = proposed_operation
        prompt_dict["reason"] = reason
        negative_trajectory_df = pd.read_csv(StringIO(prompt_dict["negative_trajectory"]), sep="\t")
        bottleneck_operation = list(negative_trajectory_df[negative_trajectory_df["operation_id"] == bottleneck_operation_id]["operator(parameter)"])[0]

        for previous_operation in negative_trajectory_df[negative_trajectory_df["operation_id"] < bottleneck_operation_id]["operator(parameter)"]:
            env.run_operator(eval(previous_operation))
        prompt_dict["bottleneck_operation"] = bottleneck_operation
        prompt_dict["state_data"] = filter_dict_to_str(env.state_data)
        prompt_dict["state_data_feature"] = filter_dict_to_str(self.get_state_data_feature_function(env.global_data, env.state_data))

        # Try to provide suggestion
        self.gpt_helper.load("extract_suggestion_v2", prompt_dict)
        response = self.gpt_helper.chat()
        suggestion = extract(response, key="suggestion")
        if suggestion:
            self.gpt_helper.dump(suggestion_name)
            # Implement the new code
            heuristic_name = prompt_dict["heuristic_name"]
            origin_function_name = prompt_dict["function_name"]
            prompt_dict["suggestion"] = suggestion
            description = f"Now, based on these suggestions:\n{suggestion}\nUpdate the {origin_function_name}."
            env_summarize = prompt_dict["env_summarize"]
            output_heuristic_file = HeuristicGenerator(self.gpt_helper, self.problem).generate(heuristic_name, description, env_summarize, smoke_test)
            if output_heuristic_file:
                suggested_result = self.validation(self.validation_cases, output_heuristic_file)
                return output_heuristic_file, suggestion, suggested_result
        return None, None, None

    def refine_heuristic(
            self,
            prompt_dict: dict,
            env: BaseEnv,
            basic_heuristic_name: str,
            basic_heuristic_result: list[float],
            previous_heuristic_name: str,
            previous_heuristic_result: list[float],
            last_heuristic_name: str,
            last_heuristic_result: list[float],
            last_suggestion: str,
            suggestion_name: str,
            time_limitation: float=10,
            smoke_test: bool=True,
    ) -> tuple[str, str, float]:
        # Compare benchmark table
        benchmark_df = self.global_feature_df.copy()
        benchmark_df[basic_heuristic_name] = basic_heuristic_result
        previous_improvement = self.get_improvement(env, basic_heuristic_result, previous_heuristic_result)
        benchmark_df[previous_heuristic_name] = [f"{previous_heuristic_result[index]}({previous_improvement[index]})" for index in range(len(basic_heuristic_result))]
        last_improvement = self.get_improvement(env, basic_heuristic_result, last_heuristic_result)
        benchmark_df[last_heuristic_name] = [f"{last_heuristic_result[index]}({last_improvement[index]})" for index in range(len(basic_heuristic_result))]
        # Compare the evolved result in validation data.
        feature_num = self.global_feature_df.shape[-1]
        if env.compare(1, 2) > 0:
            compare = "lower"
        else:
            compare = "higher"
        prompt_dict["feature_num"] = feature_num
        prompt_dict["basic_heuristic_name"] = basic_heuristic_name
        prompt_dict["previous_heuristic_name"] = previous_heuristic_name
        prompt_dict["last_heuristic_name"] = last_heuristic_name
        prompt_dict["last_suggestion"] = last_suggestion
        prompt_dict["compare"] = compare
        prompt_dict["benchmark_result"] = df_to_str(benchmark_df)

        self.gpt_helper.load("refinement", prompt_dict)
        response = self.gpt_helper.chat()
        analysis_results = extract(response, key="refinement", sep="\n")
        self.gpt_helper.dump(suggestion_name)
        suggestion = None
        for analysis_result in analysis_results:
            if "code adjustment suggestion" in analysis_result:
                suggestion = analysis_result.split(":")[-1]

        if suggestion:
            # Implement the new code
            heuristic_name = prompt_dict["heuristic_name"]
            prompt_dict["suggestion"] = suggestion
            description = f"Now, based on these suggestions:\n{suggestion}\nUpdate the {last_heuristic_name}."
            env_summarize = prompt_dict["env_summarize"]
            output_heuristic_file = HeuristicGenerator(self.gpt_helper, self.problem).generate(heuristic_name, description, env_summarize, smoke_test)
            if output_heuristic_file:
                suggested_heuristic_result = self.validation(self.validation_cases, output_heuristic_file)
                return output_heuristic_file, suggestion, suggested_heuristic_result
        return None, None, None

    def validation(
            self,
            validation_cases: list[str],
            heuristic_file: str,
            time_limitation: float=10,
            dump_result: bool=False
        ) -> list[float]:
        validation_results = []
        heuristic_name = heuristic_file.split(os.sep)[-1].split(".py")[0]
        for data_name in validation_cases:
            env = Env(data_name=data_name)
            env.reset(heuristic_name)
            hyper_heuristic = SingleHyperHeuristic(heuristic_file, problem=self.problem)
            is_complete_valid_solution = hyper_heuristic.run(env, time_limitation)
            result = env.key_value if is_complete_valid_solution else None
            if dump_result:
                env.dump_result()
            validation_results.append(result)
        return validation_results
    
    def get_improvement(self, env:BaseEnv, baselines: list[float], results: list[float]) -> float:
        improvements = [round(env.compare(results[index], baselines[index]) / baselines[index], 2) for index in range(len(baselines))]
        return improvements