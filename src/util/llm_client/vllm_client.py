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
        messages = self.normalize_messages_for_vllm()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,
            stream=False,
        )
        response_content = str(response.choices[-1].message.content or "") + str(response.choices[-1].message.reasoning_content or "")
        return response_content

    def chat_once_with_tools(self, tools: List[Dict] = None) -> Tuple[str, List[Tuple[str, Dict]]]:
        messages = self.normalize_messages_for_vllm()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="required",
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,
            stream=False,
        )

        tool_calls = response.choices[-1].message.tool_calls or []
        response_content = str(response.choices[-1].message.content or "") + str(response.choices[-1].message.reasoning_content or "")
        function_name_parameters = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            parameters = json.loads(tool_call.function.arguments)
            function_name_parameters.append((function_name, parameters))
        return response_content, function_name_parameters

    def chat_with_logprobs(self) -> Tuple[str, List[Dict[str, float]], List[str]]:
        messages = self.normalize_messages_for_vllm()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,
            stream=False,
            logprobs=True,
            top_logprobs=20
        )
        response_content = str(response.choices[-1].message.content or "") + str(response.choices[-1].message.reasoning_content or "")
        
        # Process logprobs
        logprobs_data = []
        generated_tokens = []
        if response.choices[-1].logprobs and response.choices[-1].logprobs.content:
            for token_logprob in response.choices[-1].logprobs.content:
                generated_tokens.append(token_logprob.token)
                # Create a dict of {token: logprob} for the top candidates at this position
                position_logprobs = {}
                if token_logprob.top_logprobs:
                    for top_lp in token_logprob.top_logprobs:
                        position_logprobs[top_lp.token] = top_lp.logprob
                else:
                    # Fallback if top_logprobs is empty but we have the chosen token
                    position_logprobs[token_logprob.token] = token_logprob.logprob
                logprobs_data.append(position_logprobs)
                
        return response_content, logprobs_data, generated_tokens
