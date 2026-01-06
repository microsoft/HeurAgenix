from typing import Tuple, List, Dict
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
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def _format_messages(self, messages: List[Dict]) -> List[Dict]:
        format_messages = []
        for m in messages:
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
        return format_messages

    def chat_once(self) -> str:
        format_messages = self._format_messages(self.messages)

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

    def get_sequence_score(self, conversation: List[Dict], response: str) -> float:
        format_messages = self._format_messages(conversation)

        # Apply chat template to get the prompt part
        prompt_text = self.pipeline.tokenizer.apply_chat_template(
            format_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.think,
        )
        
        # Tokenize prompt and full text (prompt + choice)
        prompt_ids = self.pipeline.tokenizer(prompt_text, return_tensors="pt").input_ids
        choice_ids = self.pipeline.tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids
        
        # Concatenate prompt and choice
        input_ids = torch.cat([prompt_ids, choice_ids], dim=1)
        
        if hasattr(self.pipeline.model, "device"):
            input_ids = input_ids.to(self.pipeline.model.device)

        # We only need to compute loss for the choice part
        # Labels are input_ids, but we mask the prompt part with -100
        labels = input_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100
        
        with torch.no_grad():
            outputs = self.pipeline.model(input_ids, labels=labels)
            # The loss returned is the average NLL over the unmasked tokens (choice_text)
            nll = outputs.loss.item()
            
        return nll

