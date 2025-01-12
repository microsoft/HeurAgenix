import copy
import math
import traceback
from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic
from src.problems.base.components import BaseOperator
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic, extract_function_with_short_docstring, extract, filter_dict_to_str, search_file
from src.util.gpt_helper_debug import GPTHelper


class GPTDeepSelectionHyperHeuristic:
    def __init__(
        self,
        gpt_helper: GPTHelper,
        heuristic_pool: list[str],
        problem: str,
    ) -> None:
        self.gpt_helper = gpt_helper
        self.problem = problem
        self.heuristic_pool = heuristic_pool
        self.heuristic_pools = {
            heuristic_file.split(".")[0]: load_heuristic(heuristic_file, problem=self.problem)
            for heuristic_file in heuristic_pool}
        self.get_global_data_feature_function = load_heuristic("evaluation_function.py", problem=self.problem, function_name="get_global_data_feature")
        self.get_state_data_feature_function = load_heuristic("evaluation_function.py", problem=self.problem, function_name="get_state_data_feature")

    def run(self, env:BaseEnv, max_steps: int=None, data_feature_content_threshold: int=1000, search_interval: int=None, search_time: int=None, **kwargs) -> bool:
        max_steps = max_steps if max_steps is not None else env.construction_steps * 3
        search_interval = search_interval if search_interval is not None else math.floor(env.construction_steps / 20)
        search_time = search_time if search_time is not None else math.floor(env.construction_steps / 10)

        search_interval = 5
        search_time = 40

        # Load background
        prompt_dict = self.gpt_helper.load_background(self.problem)

        # Classify heuristic pool
        prompt_dict["heuristic_pool_introduction"] = "\n".join([open(search_file(heuristic_file, self.problem)).read() for heuristic_file in self.heuristic_pool])
        self.gpt_helper.load("heuristic_classification", prompt_dict)
        response = self.gpt_helper.chat()
        heuristic_classification = extract(response, "heuristic_classification", sep="\n")
        classified_heuristic = {}
        classified_heuristic_introduction = ""
        heuristic_pool_names = []
        for classification in heuristic_classification:
            classification_name = classification.split(":")[0]
            heuristic_names = classification.split(":")[1].split(";")[0].split(",")
            heuristic_pool_names.extend(heuristic_names)
            reason = classification.split(":")[1].split(";")[-1]
            classified_heuristic[classification_name] = []
            classified_heuristic_introduction += f"\n{classification_name}: {reason}\n"
            for heuristic_name in heuristic_names:
                classified_heuristic[classification_name].append(self.heuristic_pools[heuristic_name])
                classified_heuristic_introduction += extract_function_with_short_docstring(open(search_file(heuristic_name + ".py", self.problem)).read(), heuristic_name) + "\n"
        self.gpt_helper.dump("heuristic_classification")
        random_hh = RandomHyperHeuristic(heuristic_pool=heuristic_pool_names, problem=self.problem)

        # Make plan
        self.gpt_helper.load_chat("background")
        global_data_feature = self.get_global_data_feature_function(env.global_data)
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
        best_result = None
        while current_steps <= max_steps and env.continue_run:
            try:
                # Select heuristic category
                self.gpt_helper.load_chat("make_plan")
                state_data_feature = self.get_state_data_feature_function(env.global_data, env.state_data)
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
                state_data_feature = self.get_state_data_feature_function(env.global_data, env.state_data)
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
                    
                    # MCTS for best heuristic in target heuristic category
                    pre_status = env.get_observation()
                    last_step_env = copy.deepcopy(env)
                    best_average_score = None
                    best_heuristic = None
                    for heuristic in candidate_heuristics:
                        # Run current heuristic for search interval times
                        last_step_after_heuristic_env = copy.deepcopy(last_step_env)
                        for _ in range(search_interval):
                            last_step_after_heuristic_env.run_heuristic(heuristic)
                        results = []
                        # MCTS to evaluate heuristic performance
                        for _ in range(search_time):
                            random_mcts_env = copy.deepcopy(last_step_after_heuristic_env)
                            complete_and_valid_solution = random_hh.run(random_mcts_env, max_steps=int(env.construction_steps*2))
                            if complete_and_valid_solution:
                                results.append(random_mcts_env.key_value)
                                # Save best
                                if best_result is None or env.compare(random_mcts_env.key_value, best_result) >= 0:
                                    random_mcts_env.dump_result(dump_trajectory=True)
                                    best_result = random_mcts_env.key_value
                        # Compare result
                        if len(results) > 0:
                            average_score = sum(results) / len(results)
                            # Select best heuristic
                            if not best_average_score or env.compare(average_score, best_average_score) > 0:
                                best_average_score = average_score
                                best_heuristic = heuristic
                                best_heuristic_env = copy.deepcopy(last_step_after_heuristic_env)

                    # Restore best heuristic
                    if not best_heuristic:
                        env = copy.deepcopy(last_step_env)
                    else:
                        env = copy.deepcopy(best_heuristic_env)

                    # Record
                    cur_status = env.get_observation()
                    heuristic_dict = {
                        "Heuristic": best_heuristic.__name__,
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
                chat_index += 1
            except Exception as e:
                trace_string = traceback.format_exc()
                print(trace_string)
        
        return env.is_complete_solution and env.is_valid_solution
