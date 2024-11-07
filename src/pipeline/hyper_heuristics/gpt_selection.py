import os
import traceback
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic, extract_function_with_short_docstring, extract, filter_dict_to_str
from src.util.gpt_helper import GPTHelper


class GPTSelectionHyperHeuristic:
    def __init__(
        self,
        gpt_helper: GPTHelper,
        problem: str="tsp",
        heuristic_dir: str=None
    ) -> None:
        self.gpt_helper = gpt_helper
        self.problem = problem
        self.heuristic_dir = heuristic_dir if heuristic_dir is not None else os.path.join("src", "problems", problem, "heuristics", "basic_heuristics")
        heuristic_names = [heuristic_file.split(".")[0] for heuristic_file in os.listdir(self.heuristic_dir)]
        self.heuristic_docs = {heuristic_name: extract_function_with_short_docstring(open(os.path.join(self.heuristic_dir, heuristic_name + ".py")).read(), heuristic_name) for heuristic_name in heuristic_names}
        self.heuristic_pools = {heuristic_name: load_heuristic(os.path.join(self.heuristic_dir, heuristic_name + ".py")) for heuristic_name in heuristic_names}
        if os.path.exists(os.path.join("output", problem, "evaluation_function.py")):
            evaluation_function_file = os.path.join("output", problem, "evaluation_function.py")
        elif os.path.exists(os.path.join("output", problem, "generate_evaluation_function", "evaluation_function.py")):
            evaluation_function_file = os.path.join("output", problem, "generate_evaluation_function", "evaluation_function.py")
        elif os.path.exists(os.path.join("src", "problems", problem, "evaluation_function.py")):
            evaluation_function_file = os.path.join("src", "problems", problem, "evaluation_function.py")
        self.get_global_data_feature_function = load_heuristic(evaluation_function_file, "get_global_data_feature")
        self.get_state_data_feature_function = load_heuristic(evaluation_function_file, "get_state_data_feature")

    def run(self, env:BaseEnv, max_steps: int=None, validation: bool=True, data_feature_content_threshold: int=1000, **kwargs) -> None:
        # Load background
        prompt_dict = self.gpt_helper.load_background(self.problem)

        # Load heuristic pool
        max_steps = max_steps if max_steps is not None else env.construction_steps * 2
        prompt_dict["heuristic_pool_introduction"] = "\n".join(self.heuristic_docs.values())
        self.gpt_helper.load("heuristic_pool", prompt_dict)
        self.gpt_helper.chat()
        self.gpt_helper.dump("heuristic_pool")

        # Generate global heuristic value
        global_data = env.global_data
        global_data_feature = self.get_global_data_feature_function(global_data)
        prompt_dict["global_data_feature"] = filter_dict_to_str([global_data, global_data_feature], data_feature_content_threshold)

        heuristic_traject = []
        current_steps = 0
        while current_steps <= max_steps or not env.is_complete_solution:
            try:
                self.gpt_helper.load_chat("heuristic_pool")

                # Generate state heuristic value
                state_data = env.state_data
                state_data_feature = self.get_state_data_feature_function(global_data, state_data)
                prompt_dict["state_data_feature"] = filter_dict_to_str([state_data, state_data_feature], data_feature_content_threshold)

                # Generate trajectory
                if heuristic_traject == []:
                    heuristic_trajectory_str = "None"
                    last_heuristic = "None"
                else:
                    heuristic_trajectory_str = "\n".join([f"---Round {round}---\n" + "\n".join(f"{key}: {value}" for key, value in items.items()) for round, items in enumerate(heuristic_traject)])
                    last_heuristic = heuristic_traject[-1]["Heuristic"]
                prompt_dict["discuss_round"] = str(len(heuristic_traject))
                prompt_dict["heuristic_traject"] = heuristic_trajectory_str
                prompt_dict["last_heuristic"] = last_heuristic
                state_data_feature = self.get_state_data_feature_function(env.global_data, env.state_data)
                state_data_feature.update(env.state_data)
                for key, value in global_data_feature.items():  
                    if len(str(key) + str(value)) <= data_feature_content_threshold:  
                        prompt_dict[key] = value
                        prompt_dict.update(env.global_data)
                self.gpt_helper.load("heuristic_selection", prompt_dict)

                response = self.gpt_helper.chat()
                self.gpt_helper.dump(f"step_{len(heuristic_traject)}")

                if "Run heuristic:" in response:
                    # Load selected heuristic, running step, parameters(optional) and reason
                    result = extract(response, key="Run heuristic", sep="\n")
                    selected_heuristic_name = result[0].split(":")[-1].strip()
                    running_step = int(result[1].split(":")[-1].strip().split(" ")[0])
                    explain = result[-1].split(":")[-1].strip()
                    parameters = {}
                    if len(result) > 3:
                        try:
                            parameter_str = result[2].split(": ")[-1]
                            parameter_str = "NA"
                            pairs = [pair.split("=") for pair in parameter_str.split(";")]
                            parameters = {key: float(value) if '.' in value else int(value) if value.isdigit() else value for key, value in pairs}
                        except Exception as e:
                            pass

                    assert selected_heuristic_name in self.heuristic_pools.keys()
                    selected_heuristic = self.heuristic_pools[selected_heuristic_name]

                    pre_status = env.get_observation()
                    for _ in range(running_step):
                        env.run_heuristic(selected_heuristic, parameters=parameters, validation=validation)
                    cur_status = env.get_observation()
                    heuristic_dict = {
                        "Heuristic": selected_heuristic_name,
                        "Parameters": parameters,
                        "Running Steps": running_step,
                        "Explain": explain
                    }
                    for key in pre_status.keys():
                        heuristic_dict["Delta of " + key] = cur_status[key] - pre_status[key]
                    heuristic_traject.append(heuristic_dict)
                    current_steps += running_step
                    if env.is_complete_solution:
                        env.dump_result()
                elif "Stop" in response or "None" in response:
                    if env.is_complete_solution:
                        break
                    else:
                        current_steps -= 1
            except Exception as e:
                trace_string = traceback.format_exc()
                print(trace_string)
        return env.is_complete_solution
