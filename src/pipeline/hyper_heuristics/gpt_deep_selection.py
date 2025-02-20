import os
import math
import traceback
import multiprocessing
import multiprocessing.managers
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic, extract_function_with_short_docstring, extract, filter_dict_to_str, search_file
from src.util.gpt_helper import GPTHelper
from src.util.compare_heuristics import compare_heuristics


class GPTDeepSelectionHyperHeuristic:
    def __init__(
        self,
        gpt_helper: GPTHelper,
        heuristic_pool: list[str],
        problem: str,
        search_interval: int=None,
        search_time: int=None,
    ) -> None:
        self.gpt_helper = gpt_helper
        self.problem = problem
        self.search_interval = search_interval
        self.search_time = search_time
        self.heuristic_pool = [heuristic_name.split(os.sep)[-1].split(".")[0] for heuristic_name in heuristic_pool]

    def run(self, env:BaseEnv, max_steps: int=None, data_feature_content_threshold: int=1000, **kwargs) -> bool:
        # Init parameters
        max_steps = max_steps if max_steps is not None else env.construction_steps * 3
        running_max_steps = env.construction_steps * 2
        search_interval = self.search_interval if self.search_interval is not None else math.floor(env.construction_steps / 20)
        search_time = self.search_time if self.search_time is not None else math.floor(env.construction_steps / 10)

        # Init feature function
        get_global_data_feature_function = load_heuristic("evaluation_function.py", problem=self.problem, function_name="get_global_data_feature")
        get_state_data_feature_function = load_heuristic("evaluation_function.py", problem=self.problem, function_name="get_state_data_feature")
        

        # Load background
        prompt_dict = self.gpt_helper.load_background(self.problem)

        # Classify heuristic pool
        prompt_dict["heuristic_pool_introduction"] = "\n".join([open(search_file(heuristic_file + ".py", self.problem)).read() for heuristic_file in self.heuristic_pool])
        self.gpt_helper.load("heuristic_classification", prompt_dict)
        response = self.gpt_helper.chat()
        heuristic_classification = extract(response, "heuristic_classification", sep="\n")
        classified_heuristic = {}
        classified_heuristic_introduction = ""
        running_heuristic_pool = []
        for classification in heuristic_classification:
            classification_name = classification.split(":")[0]
            heuristic_names = classification.split(":")[1].split(";")[0].split(",")
            running_heuristic_pool.extend(heuristic_names)
            reason = classification.split(":")[1].split(";")[-1]
            classified_heuristic[classification_name] = []
            classified_heuristic_introduction += f"\n{classification_name}: {reason}\n"
            for heuristic_name in heuristic_names:
                assert heuristic_name in self.heuristic_pool
                classified_heuristic[classification_name].append(heuristic_name)
                classified_heuristic_introduction += extract_function_with_short_docstring(open(search_file(heuristic_name + ".py", self.problem)).read(), heuristic_name) + "\n"
        self.gpt_helper.dump("heuristic_classification")

        # Make plan
        self.gpt_helper.load_chat("background")
        global_data_feature = get_global_data_feature_function(env.global_data)
        prompt_dict["global_data_feature"] = filter_dict_to_str([env.global_data, global_data_feature], data_feature_content_threshold)
        prompt_dict["category_names"] = ",".join(classified_heuristic.keys())
        prompt_dict["classified_heuristic_introduction"] = classified_heuristic_introduction
        prompt_dict["running_step"] = search_interval
        self.gpt_helper.load("make_plan", prompt_dict)
        self.gpt_helper.chat()
        self.gpt_helper.dump("make_plan")

        heuristic_traject = []
        current_steps = 0
        chat_index = 0
        manager = multiprocessing.Manager()
        best_result_proxy = manager.Value('d', float('-inf'))
        while current_steps <= max_steps and env.continue_run:
            try:
                # Select heuristic category
                self.gpt_helper.load_chat("make_plan")
                state_data_feature = get_state_data_feature_function(env.global_data, env.state_data)
                prompt_dict["state_data_feature"] = filter_dict_to_str([env.state_data, state_data_feature], data_feature_content_threshold)
                if heuristic_traject == []:
                    heuristic_trajectory_str = "None"
                    last_heuristic = "None"
                    last_heuristic_category = "None"
                else:
                    heuristic_trajectory_str = "\n".join([f"---Round {round}---\n" + "\n".join(f"{key}: {value}" for key, value in items.items()) for round, items in enumerate(heuristic_traject)])
                    last_heuristic = heuristic_traject[-1]["Heuristic"]
                    last_heuristic_category = heuristic_traject[-1]["Heuristic Category"]

                prompt_dict["discuss_round"] = chat_index
                prompt_dict["heuristic_traject"] = heuristic_trajectory_str
                prompt_dict["last_heuristic"] = last_heuristic
                prompt_dict["last_heuristic_category"] = last_heuristic_category
                state_data_feature = get_state_data_feature_function(env.global_data, env.state_data)
                state_data_feature.update(env.state_data)
                for key, value in global_data_feature.items():  
                    if len(str(key) + str(value)) <= data_feature_content_threshold:  
                        prompt_dict[key] = value
                        prompt_dict.update(env.global_data)

                self.gpt_helper.load("heuristic_category_selection", prompt_dict)
                response = self.gpt_helper.chat()
                self.gpt_helper.dump(f"step_{chat_index}")

                # Continue run
                if "continue_run" in response or "None" in response:
                    # Extract target heuristic category
                    result = extract(response, key="continue_run", sep="\n")
                    selected_heuristic_category = result[0].split(":")[-1]
                    explain = result[1].split(":")[-1]
                    assert selected_heuristic_category in classified_heuristic
                    candidate_heuristics = classified_heuristic[selected_heuristic_category]
                    
                    # Evaluate each heuristics by sample
                    pre_status = env.get_observation()
                    total_results = compare_heuristics(
                        env,
                        candidate_heuristics,
                        running_heuristic_pool,
                        running_max_steps,
                        search_interval,
                        search_time,
                        self.problem,
                        best_result_proxy,
                        True
                    )

                    # Find the best heuristic
                    best_heuristic_name = None
                    best_average_score = None
                    best_after_heuristic_env = None
                    for heuristic, results, after_step_env, operators in total_results:
                        average_score = None if len(results) <= 0 else sum(results) / len(results)
                        if average_score is not None and best_average_score is None or env.compare(average_score, best_average_score) > 0:
                            best_heuristic_name = heuristic
                            best_average_score = average_score
                            best_after_heuristic_env = after_step_env
                    
                    for attr in vars(env):
                        setattr(env, attr, getattr(best_after_heuristic_env, attr))

                    # Record
                    cur_status = env.get_observation()
                    heuristic_dict = {
                        "Heuristic": best_heuristic_name,
                        "Heuristic Category": selected_heuristic_category,
                        "Running Steps": search_interval,
                        "Explain": explain
                    }
                    for key in pre_status.keys():
                        heuristic_dict[key] = f"{pre_status[key]}->{cur_status[key]}"
                    heuristic_traject.append(heuristic_dict)
                    current_steps += search_interval
                elif "Stop" in response or "None" in response:
                    # Check to stop
                    if env.is_complete_solution:
                        break
                    else:
                        chat_index -= 1
                chat_index += 1
            except Exception as e:
                trace_string = traceback.format_exc()
                print(trace_string)
        return env.is_complete_solution and env.is_valid_solution
