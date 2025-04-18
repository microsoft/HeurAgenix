import os
import json
import re
import base64
import importlib
from time import sleep
from src.util.util import compress_numbers, extract, load_framework_description


class BaseLLMClient:
    def __init__(
            self,
            config: dict,
            prompt_dir: str=None,
            output_dir: str=None,
        ):
        self.prompt_dir = prompt_dir
        self.output_dir = output_dir
        self.config = config
        self.reset(output_dir)

    def reset(self, output_dir:str=None) -> None:
        self.messages = []
        if output_dir is not None:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

    def chat_once(self) -> str:
        pass

    def chat(self) -> str:
        for index in range(self.max_attempts):
            try:
                response_content = self.chat_once()
                self.messages.append({"role": "assistant", "content": [{"type": "text", "text": response_content}]})
                return response_content
            except Exception as e:
                print(f"Try to chat {index + 1} time: {e}")
                sleep_time = self.sleep_time
                sleep(sleep_time)
        self.messages.append({"role": "assistant", "content": "Exceeded the maximum number of attempts"})
        self.dump("error")

    def load_chat(self, chat_file: str) -> None:
        if chat_file.split(".")[-1] != "json":
            chat_file = chat_file + ".json"
        if self.prompt_dir is not None and os.path.exists(os.path.join(self.prompt_dir, chat_file)):
            chat_file = os.path.join(self.prompt_dir, chat_file)
        elif self.prompt_dir is not None and os.path.exists(os.path.join(self.output_dir, chat_file)):
            chat_file = os.path.join(self.output_dir, chat_file)
        with open(chat_file, "r") as fp:
            self.messages = json.load(fp)

    def load_background(self, problem: str, reference_data: str=None) -> dict:
        # Load background
        problem_dir = os.path.join("src", "problems", problem, "prompt")
        if os.path.exists(os.path.join("src", "problems", problem, "components.py")):
            component_code = open(os.path.join("src", "problems", problem, "components.py")).read()
        else:
            component_code = open(os.path.join("src", "problems", "base", "mdp_components.py")).read()
        solution_class_str, operator_class_str = load_framework_description(component_code)

        env_summarize = "All data is possible"
        if reference_data:
            module = importlib.import_module(f"src.problems.{problem}.env")
            globals()["Env"] = getattr(module, "Env")
            env = Env(reference_data)
            env_summarize = env.summarize_env()

        prompt_dict = {
            "problem": problem,
            "problem_description": open(os.path.join(problem_dir, "problem_description.txt"), encoding="utf-8").read(),
            "global_data_introduction": open(os.path.join(problem_dir, "global_data.txt"), encoding="utf-8").read(),
            "state_data_introduction": open(os.path.join(problem_dir, "state_data.txt"), encoding="utf-8").read(),
            "solution_class": solution_class_str,
            "operator_class": operator_class_str,
            "env_summarize": env_summarize
        }

        self.load("background", prompt_dict)
        response = self.chat()
        is_cop = extract(response, "is_cop", "\n")
        self.dump("background")
        if not is_cop or "no" in is_cop or "No" in is_cop or "NO" in is_cop:
            raise BaseException("Not combination optimization problem")
        return prompt_dict

    def load(self, message: str, replace: dict={}) -> None:
        if self.prompt_dir is not None and os.path.exists(os.path.join(self.prompt_dir, message)):
            message = open(os.path.join(self.prompt_dir, message), "r", encoding="UTF-8").read()
        elif self.prompt_dir is not None and os.path.exists(os.path.join(self.prompt_dir, message)):
            message = open(os.path.join(self.prompt_dir, message), "r", encoding="UTF-8").read()
        elif self.prompt_dir is not None and os.path.exists(os.path.join(self.prompt_dir, message + ".txt")):
            message = open(os.path.join(self.prompt_dir, message + ".txt"), "r", encoding="UTF-8").read()
        elif os.path.exists(message):
            message = open(message, "r", encoding="UTF-8").read()
        elif os.path.exists(message + ".txt"):
            message = open(message + ".txt", "r", encoding="UTF-8").read()
        for key, value in replace.items():
            if value is None or str(value) == "":
                value = "None"
            message = message.replace("{" + key + "}", str(value))
        image_key = r"\[image: (.*?)\]"
        texts = re.split(image_key, message)
        images = re.compile(image_key).findall(message)
        current_message = []
        for i in range(len(texts)):
            if i % 2 == 1:
                encoded_image = base64.b64encode(open(images[int((i - 1)/ 2)], 'rb').read()).decode('ascii')
                current_message.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    "image_path": images[int((i - 1)/ 2)]
                })
            else:
                current_message.append({
                    "type": "text",
                    "text": compress_numbers(texts[i])
                })
        self.messages.append({"role": "user", "content": current_message})

    def dump(self, output_name: str=None) -> str:
        if self.output_dir != None and output_name != None:
            json_output_file = os.path.join(self.output_dir, f"{output_name}.json")
            text_output_file = os.path.join(self.output_dir, f"{output_name}.txt")
            print(f"Chat dumped to {text_output_file}")
            with open(json_output_file, "w") as fp:
                json.dump(self.messages, fp, indent=4)

            with open(text_output_file, "w", encoding="UTF-8") as file:
                for message in self.messages:
                    file.write(message["role"] + "\n")
                    contents = ""
                    for i, content in enumerate(message["content"]):
                        if content["type"] == "image_url":
                            contents += f"[image: {content['image_path']}]"
                        else:
                            contents += content["text"]
                    file.write(contents + "\n------------------------------------------------------------------------------------\n\n")
        return self.messages[-1]["content"][0]["text"]
