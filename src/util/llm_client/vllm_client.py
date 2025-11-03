
from typing import List, Dict, Tuple
import os
import json
from openai import OpenAI
from src.util.llm_client.base_llm_client import BaseLLMClient

class VLLMClient(BaseLLMClient):
    def __init__(
            self,
            config: dict,
            prompt_dir: str=None,
            output_dir: str=None,
        ):
        super().__init__(config, prompt_dir, output_dir)

        self.base_url = config.get("base_url", "http://localhost:8000/v1")
        if "model" in config:
            self.model = config.get("model")
        elif "model_path" in config:
            self.model = config.get("model_path")
        else:
            raise Exception("No model or model_path in config")

        self.top_p = config.get("top-p", 0.7)
        self.temperature = config.get("temperature", 0.95)
        self.max_tokens = config.get("max_tokens", 1024)
        self.seed = config.get("seed", None)
        self.max_attempts = config.get("max_attempts", 10)
        self.sleep_time = config.get("sleep_time", 2)

        api_key = config.get("api_key", "EMPTY")
        self.client = OpenAI(base_url=self.base_url, api_key=api_key)

    def reset(self, output_dir: str=None) -> None:
        self.messages = []
        if output_dir is not None:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

    def normalize_messages_for_vllm(self):
        norm = []
        for m in self.messages:
            c = m.get("content", "")
            if isinstance(c, list):
                text_parts = []
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                c = "\n".join(text_parts)
            elif not isinstance(c, str):
                c = str(c)
            norm.append({"role": m["role"], "content": c, **{k:v for k,v in m.items() if k not in ["role","content"]}})
        return norm

    def chat_once(self) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self.normalize_messages_for_vllm(),
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,
            stream=False,
        )
        msg = resp.choices[-1].message
        return msg.content or ""

    def chat_once_with_tools(self, tools: List[Dict] = None) -> List[Tuple[str, Dict]]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.normalize_messages_for_vllm(),
            tools=tools,
            tool_choice="required",
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,
            stream=False,
        )

        tool_calls = response.choices[-1].message.tool_calls or []
        function_name_parameters = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            parameters = json.loads(tool_call.function.arguments)
            function_name_parameters.append((function_name, parameters))
        return function_name_parameters

        

