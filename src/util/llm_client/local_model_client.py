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

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            model_kwargs={"torch_dtype": torch.bfloat16}
        )

    def chat_once(self) -> str:
        format_messages = []
        for m in self.messages:
            c = m.get("content", "")
            if isinstance(c, list):
                parts = []
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(p.get("text", ""))
                    elif isinstance(p, str):
                        parts.append(p)
                c = "\n".join(parts)
            elif not isinstance(c, str):
                c = str(c)
            format_messages.append({"role": m["role"], "content": c})

        text = self.pipeline.tokenizer.apply_chat_template(
            format_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.think,
        )
        response = self.pipeline(
            text,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            return_full_text=False,
        )
        response_content = response[0]["generated_text"].strip()
        return response_content
