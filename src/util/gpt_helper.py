import os
import json
import re
import base64
import importlib
import requests
from time import sleep
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from src.util.util import load_framework_description
import transformers
import torch

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

        self.api_type = gpt_setting.get("api_type", "azure")
        self.api_version = gpt_setting["api_version"]
        self.model = gpt_setting["model"]
        self.temperature = gpt_setting["temperature"]
        self.top_p = gpt_setting["top-p"]
        self.seed = gpt_setting.get("seed", None)
        self.max_tokens = gpt_setting["max_tokens"]
        self.max_attempts = gpt_setting["max_attempts"]
        self.default_sleep_time = gpt_setting["sleep_time"]
        self.azure_endpoint = gpt_setting["azure_endpoint"]
        self.local_model = gpt_setting["local_model"]
        
        # 本地模型端点（当 api_type = local 时使用）
        self.local_endpoint = gpt_setting.get("local_endpoint", "http://127.0.0.1:8000/generate")

        if self.api_type == "azure":
            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
            self.client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                azure_ad_token_provider=token_provider,
                api_version=self.api_version,
                max_retries=5,
            )
        elif self.api_type == "local":
                # model_id = "meta-llama/Llama-3.3-70B-Instruct"
                model_id = self.local_model
                self.pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_id,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device_map="auto",
                )
        else:
            self.client = None  # 本地模型不使用AzureOpenAI Client

        self.reset(output_dir)

    def reset(self, output_dir:str=None) -> None:
        self.current_message = []
        self.messages = []
        if output_dir is not None:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)


    def extract_text_from_json(self,input_string):
        """
        Extract the text content from a JSON-like string if it starts with '['.
        """
        if input_string.startswith('['):
            input_string = input_string[27:-3]
        return input_string


    def chat(self,ext_message = None) -> str:
        if ext_message != None:
            self.messages = ext_message
        # 将当前user输入加入到消息列表中
        if self.current_message != []:
            # TODO:此处为什么直接将这个list放进来
            self.messages.append({"role": "user", "content": self.current_message})
            self.current_message = []

        for index in range(self.max_attempts):
            try:
                if self.api_type == "azure":
                    # 使用Azure的chat接口
                    response = self.client.chat.completions.create(
                        model=self.model,
                        # reasoning_effort = "medium",
                        messages=self.messages,
                        # temperature=self.temperature,
                        max_completion_tokens=self.max_tokens,
                        # top_p=self.top_p,
                        seed=self.seed,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None,
                        stream=False,
                    )
                    response_content = response.choices[-1].message.content
                else:
                    # 使用本地模型接口
                    outputs = self.pipeline(
                        self.messages,
                        max_new_tokens = self.max_tokens,
                        temperature = self.temperature,
                        top_p = self.top_p
                    )
                    # local_response = requests.post(self.local_endpoint, json=payload)
                    response_content = outputs[0]["generated_text"][-1]['content']
                    # 提取回复正文
                    response_content = self.extract_text_from_json(response_content)
                    # # 根据本地服务的返回格式获取文本，这里假设返回包含 "generated_text" 字段
                    # response_content = local_response_json.get("generated_text", "")

                # 将assistant的回答加入消息记录中
                self.messages.append({"role": "assistant", "content": [{"type": "text", "text": response_content}]})
                # self.messages.append({"role": "assistant", "content": [response_content]})
                return response_content
            except Exception as e:
                print(f"Try to chat {index + 1} time: {e}")
                sleep_time = self.default_sleep_time
                if "Please retry after " in str(e) and " seconds." in str(e):
                    sleep_time = int(str(e).split("Please retry after ")[1].split(" seconds.")[0]) + 1
                sleep(sleep_time)
        
        # 超过max_attempts依然失败
        self.messages.append({"role": "assistant", "content": "Exceeded the maximum number of attempts"})
        self.dump("error")

    def _extract_text_from_messages(self):
        """从messages中抽取最后一轮user输入的文本内容作为prompt"""
        # 最后一条用户消息为self.messages中role为"user"的最后一个元素
        # 每条消息的content是列表，其中的内容有{"type":"text","text":...}
        # 可以将所有 "type":"text" 的内容拼接为最终prompt
        user_prompts = []
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                for c in msg["content"]:
                    if c["type"] == "text":
                        user_prompts.append(c["text"])
                break
        return "\n".join(user_prompts)

    def load_chat(self, chat_file: str) -> None:
        if chat_file.split(".")[-1] != "json":
            chat_file = chat_file + ".json"
        if self.prompt_dir is not None and os.path.exists(os.path.join(self.prompt_dir, chat_file)):
            chat_file = os.path.join(self.prompt_dir, chat_file)
        elif self.prompt_dir is not None and os.path.exists(os.path.join(self.output_dir, chat_file)):
            chat_file = os.path.join(self.output_dir, chat_file)
        with open(chat_file, "r") as fp:
            self.messages = json.load(fp)

    def load_prompt_dict(self, problem: str, reference_data: str=None) -> dict:
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
            "problem_description": open(os.path.join(problem_dir, "problem_description.txt")).read(),
            "global_data_introduction": open(os.path.join(problem_dir, "global_data.txt")).read(),
            "state_data_introduction": open(os.path.join(problem_dir, "state_data.txt")).read(),
            "solution_class": solution_class_str,
            "operator_class": operator_class_str,
            "env_summarize": env_summarize
        }

        
        return prompt_dict
    
    def load_heuristic(self, problem: str, reference_data: str=None) -> dict:
        problem_dir = os.path.join("src", "problems", problem, "prompt")
        if os.path.exists(os.path.join(problem_dir, "heuristic_pool.txt")):
            heuristic_pool = open(os.path.join("src", "problems", problem, "heuristic_pool.txt")).read()
        else:
            raise FileNotFoundError(f"The required file 'heuristic_pool.txt' is missing.")
        
        return heuristic_pool


    def load(self, message: str, replace: dict={}) -> None:
        if self.prompt_dir is not None and os.path.exists(os.path.join(self.prompt_dir, message)):
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
        
        # print(f'####### message:{message}')
        # print(f'####### message str over')

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

    def dump(self, output_name: str=None) -> str:
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
        return self.messages[-1]["content"][0]["text"]
