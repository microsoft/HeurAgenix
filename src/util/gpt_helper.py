import os
import json
import re
import base64
from openai import AzureOpenAI
from time import sleep
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from src.util.util import load_framework_description


class GPTHelper:
    def __init__(
            self,
            prompt_dir: str=None,
            output_dir: str=None,
            gpt_setting: dict=None,
        ):
        self.prompt_dir = prompt_dir
        self.output_dir = output_dir
        if gpt_setting is None:
            gpt_setting = json.load(open("gpt_setting.json"))

        self.api_version = gpt_setting["api_version"]
        self.model = gpt_setting["model"]
        self.temperature = gpt_setting["temperature"]
        self.top_p = gpt_setting["top-p"]
        self.seed = gpt_setting.get("seed", None)
        self.max_tokens = gpt_setting["max_tokens"]
        self.max_attempts = gpt_setting["max_attempts"]
        self.default_sleep_time = gpt_setting["sleep_time"]
        self.azure_endpoint = gpt_setting["azure_endpoint"]

        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            azure_ad_token_provider=token_provider,
            api_version=self.api_version,
            max_retries=5,
        )

        self.reset(output_dir)

    def reset(self, output_dir:str=None) -> None:
        self.current_message = []
        self.messages = []
        if output_dir is not None:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

    def chat(self) -> str:
        if self.current_message != []:
            self.messages.append({"role": "user", "content": self.current_message})
            self.current_message = []

        for index in range(self.max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    seed=self.seed,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    stream=False,
                )
                response_content = response.choices[-1].message.content
                self.messages.append({"role": "assistant", "content": [{"type": "text", "text": response_content}]})
                return response_content
            except Exception as e:
                print(f"Try to chat {index + 1} time: {e}")
                sleep_time = self.default_sleep_time
                if "Please retry after " in str(e) and " seconds." in str(e):
                    sleep_time = int(str(e).split("Please retry after ")[1].split(" seconds.")[0]) + 1
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

    def load_background(self, problem: str) -> dict:
        # Load background
        problem_dir = os.path.join("src", "problems", problem, "prompt")
        component_code = open(os.path.join("src", "problems", problem, "components.py")).read()
        solution_class_str, operator_class_str = load_framework_description(component_code)

        prompt_dict = {
            "problem": problem,
            "problem_description": open(os.path.join(problem_dir, "problem_description.txt")).read(),
            "global_data_introduction": open(os.path.join(problem_dir, "global_data.txt")).read(),
            "state_data_introduction": open(os.path.join(problem_dir, "state_data.txt")).read(),
            "solution_class": solution_class_str,
            "operator_class": operator_class_str
        }

        self.load("background", prompt_dict)
        self.chat()
        self.dump("background")
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
        for i in range(len(texts)):
            if i % 2 == 1:
                encoded_image = base64.b64encode(open(images[int((i - 1)/ 2)], 'rb').read()).decode('ascii')
                self.current_message.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    "image_path": images[int((i - 1)/ 2)]
                })
            else:
                self.current_message.append({
                    "type": "text",
                    "text": texts[i]
                })

    def dump(self, output_name: str=None) -> None:
        if self.output_dir != None and output_name != None:
            if self.current_message != []:
                self.messages.append({"role": "user", "content": self.current_message})
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
        else:
            print(self.messages[-1]["content"][0]["text"])
