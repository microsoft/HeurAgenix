import os
import ast
import transformers
import torch
from src.util.llm_client.base_llm_client import BaseLLMClient


class LocalModelClient(BaseLLMClient):
    def __init__(
            self,
            config: dict,
            prompt_dir: str=None,
            output_dir: str=None,
        ):
        super().__init__(config, prompt_dir, output_dir)

        if os.getenv("AMLT_DATA_DIR"):
            self.model = os.path.join(os.getenv("AMLT_DATA_DIR"), os.path.normpath(config['model_path']))
        else:
            self.model = os.path.normpath(config['model_path'])
        self.top_p = config.get("top-p", 0.7)
        self.temperature = config.get("temperature", 0.95)
        self.max_tokens = config.get("max_tokens", 3200)
        self.seed = config.get("seed", None)
        self.max_attempts = config.get("max_attempts", 50)
        self.sleep_time = config.get("sleep_time", 60)

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            model_kwargs={"torch_dtype": torch.bfloat16}
        )

    def chat_once(self) -> str:
        format_messages = []
        for message in self.messages:
            format_messages.append({
                "role": message["role"],
                "content": message["content"][0]["text"]
            })
        response = self.pipeline(
            format_messages,
            max_new_tokens = self.max_tokens,
            temperature = self.temperature,
            top_p = self.top_p
        )
        response_content = response[0]["generated_text"][-1]['content']
        return response_content
