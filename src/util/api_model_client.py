import os
import json
import requests
from src.util.base_llm_client import BaseLLMClient


class APIModelClient(BaseLLMClient):
    def __init__(
            self,
            prompt_dir: str=None,
            output_dir: str=None,
            setting_file: dict=None,
        ):
        if not setting_file:
            setting_file = os.path.join("src", "util", "api_model_setting.json")
        super().__init__(prompt_dir, output_dir, setting_file)

        self.url = self.setting["url"]
        model = self.setting["model"]
        stream = self.setting.get("stream", False)
        top_p = self.setting.get("top-p", 0.7)
        temperature = self.setting.get("temperature", 0.95)
        max_tokens = self.setting.get("max_tokens", 3200)
        seed = self.setting.get("seed", None)
        api_key = self.setting["api_key"]
        self.max_attempts = self.setting.get("max_attempts", 50)
        self.sleep_time = self.setting.get("sleep_time", 60)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.payload = {
            "model": model,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed
        }


    def reset(self, output_dir:str=None) -> None:
        self.messages = []
        if output_dir is not None:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

    def chat_once(self) -> str:
        self.payload["messages"] = self.messages
        response = requests.request("POST", self.url, json=self.payload, headers=self.headers)
        response_content = json.loads(response.text)["choices"][-1]["message"]["content"]
        return response_content
