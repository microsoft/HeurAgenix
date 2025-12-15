import os
import json
import requests
from src.util.llm_client.base_llm_client import BaseLLMClient


class APIModelClient(BaseLLMClient):
    def __init__(
            self,
            config: dict,
            prompt_dir: str=None,
            output_dir: str=None,
        ):
        super().__init__(config, prompt_dir, output_dir)

        self.url = config["url"]
        model = config["model"]
        stream = config.get("stream", False)
        api_key = config["api_key"]
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.payload = {
            "model": model,
            "stream": stream,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed
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
