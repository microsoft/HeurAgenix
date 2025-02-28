import os
import traceback
from src.problems.base.env import BaseEnv
from src.util.util import load_heuristic, extract_function_with_short_docstring, extract, filter_dict_to_str
from src.util.gpt_helper import GPTHelper
import re

class GPTSelectionHyperHeuristic:
    def __init__(
        self,
        gpt_helper: GPTHelper,
        heuristic_dir: str,
        problem: str,
    ) -> None:
        self.gpt_helper = gpt_helper
        self.problem = problem
        self.heuristic_docs = {
            heuristic_file.split(".")[0]: extract_function_with_short_docstring(open(os.path.join(heuristic_dir, heuristic_file)).read(), heuristic_file.split(".")[0]) 
            for heuristic_file in os.listdir(heuristic_dir)}
        self.heuristic_pools = {
            heuristic_file.split(".")[0]: load_heuristic(heuristic_file, problem=self.problem)
            for heuristic_file in os.listdir(heuristic_dir)}
        self.get_global_data_feature_function = load_heuristic("evaluation_function.py", problem=self.problem, function_name="get_global_data_feature")
        self.get_state_data_feature_function = load_heuristic("evaluation_function.py", problem=self.problem, function_name="get_state_data_feature")

    def run(self, env:BaseEnv, max_steps: int=None, data_feature_content_threshold: int=1000, **kwargs) -> bool:
        # # Load background
        prompt_dict = self.gpt_helper.load_prompt_dict(self.problem)
        # self.gpt_helper.load("background", prompt_dict)

        # self.gpt_helper.chat()
        # self.gpt_helper.dump("background")

        # # Load heuristic pool
        max_steps = max_steps if max_steps is not None else env.construction_steps * 2
        # prompt_dict["heuristic_pool_introduction"] = "\n".join(self.heuristic_docs.values())
        # self.gpt_helper.load("heuristic_pool", prompt_dict)
        # self.gpt_helper.chat()
        # self.gpt_helper.dump("heuristic_pool")

        # heuristic_promptt = self.gpt_helper.load_heuristic(self.problem) 
        # self.gpt_helper.load("src/problems/tsp/prompt/heuristic_pool.txt", prompt_dict)


        # self.gpt_helper.dump("heuristic_pool")

        # Generate global heuristic value
        global_data = env.global_data
        global_data_feature = self.get_global_data_feature_function(global_data)
        prompt_dict["global_data_feature"] = filter_dict_to_str([global_data, global_data_feature], data_feature_content_threshold)

        heuristic_traject = []
        current_steps = 0
        while current_steps <= max_steps or not env.is_complete_solution:
            try:
                # self.gpt_helper.load_chat("heuristic_pool")
                # self.messages = [{
                #     "type": "text",
                #     "text": heuristic_promptt
                # }]
                self.gpt_helper.reset()
                # Generate state heuristic value
                state_data = env.state_data
                state_data_feature = self.get_state_data_feature_function(global_data, state_data)
                prompt_dict["state_data_feature"] = filter_dict_to_str([state_data, state_data_feature], data_feature_content_threshold)
                total = 0
                # Generate trajectory
                if heuristic_traject == []:
                    heuristic_trajectory_str = "None"
                    last_heuristic = "None"
                else:
                    total = len(heuristic_traject)
                    # 如果轨迹超过 5 条，则只显示最后 5 条，且保留真实的轮次编号
                    if total > 5:
                        traj_to_show = enumerate(heuristic_traject[-5:], start=total - 5)
                    else:
                        traj_to_show = enumerate(heuristic_traject)
                        
                    heuristic_trajectory_str = "\n".join(
                        [f"---Round {round}---\n" + "\n".join(f"{key}: {value}" for key, value in items.items())
                        for round, items in traj_to_show]
                    )
                    last_heuristic = heuristic_traject[-1]["Heuristic"]

                prompt_dict["discuss_round"] = total
                prompt_dict["heuristic_traject"] = heuristic_trajectory_str
                prompt_dict["last_heuristic"] = last_heuristic
                state_data_feature = self.get_state_data_feature_function(env.global_data, env.state_data)
                state_data_feature.update(env.state_data)
                for key, value in global_data_feature.items():  
                    if len(str(key) + str(value)) <= data_feature_content_threshold:  
                        prompt_dict[key] = value
                        prompt_dict.update(env.global_data)
                self.gpt_helper.load("src/problems/tsp/prompt/heuristic_pool.txt", prompt_dict)

                response = self.gpt_helper.chat()
                # 将转义字符转换为实际的非转义字符
                response = response.replace("\\n", "\n")
                self.gpt_helper.dump(f"step_{len(heuristic_traject)}")

                if "Run heuristic:" in response:
                    # Load selected heuristic, running step, parameters(optional) and reason
                    # Match the selected heuristic
                    heuristic_name_match = re.search(r"selected heuristic:\s*([^\s\n]+)", response)
                    selected_heuristic_name = heuristic_name_match.group(1) if heuristic_name_match else None

                    # Match the running steps
                    running_steps_match = re.search(r"running steps:\s*(\d+)", response)
                    running_step = int(running_steps_match.group(1)) if running_steps_match else None

                    hype_parameters_match = re.search(r"hype parameter\(optional\):\s*([a-zA-Z0-9=;]*)", response)
                    if hype_parameters_match:
                        parameter_str = hype_parameters_match.group(1).strip()
                        # 如果参数字符串为空或不包含 '=' ，则认为没有超参数
                        if not parameter_str or "=" not in parameter_str:
                            parameters = {}
                        else:
                            # 对于每个以 ';' 分隔的参数对，确保包含 '=' 后再分割
                            pairs = [pair.split("=", 1) for pair in parameter_str.split(";") if "=" in pair]
                            parameters = {
                                key.strip(): float(value.strip()) if '.' in value.strip() else int(value.strip()) if value.strip().isdigit() else value.strip()
                                for key, value in pairs
                            }
                    else:
                        parameters = {}

                    # Match the explanation
                    explanation_match = re.search(r"explanation:\s*(.+?)\s*\*\*\*", response, re.DOTALL)
                    explain = explanation_match.group(1).strip() if explanation_match else None


                    # result = extract(response, key="Run heuristic", sep="\n")
                    # selected_heuristic_name = result[0].split(":")[-1].strip()
                    # running_step = int(result[1].split(":")[-1].strip().split(" ")[0])
                    # explain = result[-1].split(":")[-1].strip()
                    # parameters = {}
                    # if len(result) > 3:
                    #     try:
                    #         parameter_str = result[2].split(": ")[-1]
                    #         parameter_str = "NA"
                    #         pairs = [pair.split("=") for pair in parameter_str.split(";")]
                    #         parameters = {key: float(value) if '.' in value else int(value) if value.isdigit() else value for key, value in pairs}
                    #     except Exception as e:
                    #         pass
                    print('===============================================================')
                    print('=====================matching result===========================')
                    print(response)
                    print(f'selected_heuristic_name:[{selected_heuristic_name}]')
                    print(f'running_step:[{running_step}]')
                    print(f'parameters:[{parameters}]')
                    print('===============================================================')

                    assert selected_heuristic_name in self.heuristic_pools.keys()
                    selected_heuristic = self.heuristic_pools[selected_heuristic_name]

                    pre_status = env.get_observation()
                    for _ in range(running_step):
                    # tem_step = int((state_data_feature['unvisited_num']+state_data_feature['visited_num'])/20)
                    # for _ in range(tem_step):
                        print(f'Successfully run the heuristic:{selected_heuristic_name}')
                        env.run_heuristic(selected_heuristic, parameters=parameters)
                    cur_status = env.get_observation()
                    heuristic_dict = {
                        "Heuristic": selected_heuristic_name,
                        "Parameters": parameters,
                        "Running Steps": running_step,
                        # "Explain": explain
                    }
                    for key in pre_status.keys():
                        heuristic_dict["Delta of " + key] = cur_status[key] - pre_status[key]
                    heuristic_traject.append(heuristic_dict)
                    current_steps += running_step
                elif "Stop" in response or "None" in response:
                    if env.is_complete_solution:
                        break
                    else:
                        current_steps -= 1
            except Exception as e:
                trace_string = traceback.format_exc()
                print(trace_string)
        return env.is_complete_solution and env.is_valid_solution
