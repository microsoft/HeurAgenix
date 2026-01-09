from typing import Tuple, List, Dict
import os
import ast
import transformers
import torch
from src.util.llm_client.base_llm_client import BaseLLMClient


class LocalModelClient(BaseLLMClient):
    def __init__(
            self,
            config_path: str,
            system_prompt: str = None,
        ):
        super().__init__(config_path, system_prompt)
        
        if os.getenv("AMLT_DATA_DIR"):
            self.model = os.path.join(os.getenv("AMLT_DATA_DIR"), os.path.normpath(self.config['model_name']))
        else:
            self.model = os.path.normpath(self.config['model_name'])

        # Determine device to avoid distributed init issues on multi-GPU nodes
        device_map = "auto"
        if torch.cuda.is_available():
             # Force usage of the first GPU if available to prevent multi-gpu distributed init failures
            device_map = "cuda:0"

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
            },
            device_map=device_map,
            trust_remote_code=True,
        )

    def _merge_system_role(self, messages: List[Dict]) -> List[Dict]:
        """
        Merge system messages into the first user message.
        """
        new_messages = []
        system_content_parts = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content_parts.append(msg["content"])
            else:
                new_messages.append(msg)
        
        if not system_content_parts:
            return messages
            
        system_text = "\n\n".join(system_content_parts)
        
        # Find first user message
        for i, msg in enumerate(new_messages):
            if msg["role"] == "user":
                # Create a copy to avoid mutating original
                new_msg = msg.copy()
                new_msg["content"] = system_text + "\n\n" + msg["content"]
                new_messages[i] = new_msg
                return new_messages
        
        # If no user message found, prepend as user message (fallback)
        return [{"role": "user", "content": system_text}] + new_messages

    def _format_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Since we now enforce string-only content in BaseLLMClient.
        We can just pass messages through (or copy if needed),
        because they are already in [{"role": ..., "content": "..."}] format.
        """
        return messages

    def chat_once(self, continue_prefix: str = None) -> str:
        # Check if the last message is assistant. If so, and we want to continue, 
        # we might need to handle it specially.
        # But our agreed approach is: continue_prefix comes from outside, 
        # unrelated to self.messages structure for flexibility.
        
        format_messages = self._format_messages(self.messages)

        try:
            text = self.pipeline.tokenizer.apply_chat_template(
                format_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.think,
            )
        except Exception as e:
            # Fallback for models not supporting system role (e.g. Gemma)
            if "system" in str(e).lower() and ("role" in str(e).lower() or "support" in str(e).lower()):
                 format_messages = self._merge_system_role(format_messages)
                 text = self.pipeline.tokenizer.apply_chat_template(
                    format_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.think,
                )
            else:
                raise e
        
        # KEY CHANGE: Append prefix manually if provided
        # This bypasses the template's closing tokens for the previous turn
        if continue_prefix:
            text += continue_prefix
        
        gen_kwargs = {
            "max_new_tokens": self.max_tokens,
            "return_full_text": False,
        }
        
        # Prioritize 'do_sample' from config, otherwise infer from temperature
        do_sample = self.config.get("do_sample")
        if do_sample is None:
            if self.temperature == 0:
                do_sample = False
            else:
                do_sample = True
                
        gen_kwargs["do_sample"] = do_sample
        
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        response = self.pipeline(text, **gen_kwargs)
        if continue_prefix:
             # If we manually appended a prefix, the pipeline output *might* not include it 
             # (depends on return_full_text=False). 
             # Usually return_full_text=False returns ONLY new tokens.
             # So we should just return the new part.
             pass
             
        response_content = response[0]["generated_text"]
        # Don't strip immediately if we rely on whitespace continuity, but usually safe.
        # Although for math, if prefix ends in "The", generated " answer" (with space).
        # We'll leave it as is for now.
        return response_content

    def get_sequence_score(self, conversation: List[Dict], response: str) -> float:
        format_messages = self._format_messages(conversation)

        # Apply chat template to get the prompt part
        try:
            prompt_text = self.pipeline.tokenizer.apply_chat_template(
                format_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.think,
            )
        except Exception as e:
            # Fallback for models not supporting system role
            if "system" in str(e).lower() and ("role" in str(e).lower() or "support" in str(e).lower()):
                format_messages = self._merge_system_role(format_messages)
                prompt_text = self.pipeline.tokenizer.apply_chat_template(
                    format_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.think,
                )
            else:
                raise e
        
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

