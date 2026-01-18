from typing import List, Dict
import os
import transformers
import torch
import threading
from src.engine.llm_client.base_llm_client import BaseLLMClient


class TransformersClient(BaseLLMClient):
    def __init__(
            self,
            config: Dict,
            system_prompt: str = None,
            device_id: int = 0,
            logger=None,
        ):
        super().__init__(config, system_prompt, logger=logger)
        
        # In new config.yaml, the field is 'name' (e.g., "Qwen/Qwen3-8B")
        model_name = self.config.get('name')

        if not model_name:
             raise ValueError(f"Config missing 'name' field for model. Config: {self.config}")
         
        if os.getenv("AMLT_DATA_DIR"):
            self.model = os.path.join(os.getenv("AMLT_DATA_DIR"), os.path.normpath(model_name))
        else:
            self.model = os.path.normpath(model_name)

        self.lock = threading.Lock()

        # Determine device to avoid distributed init issues on multi-GPU nodes
        # Use integer device for strict placement. device_map can sometimes be flaky in pipelines.
        device = device_id if torch.cuda.is_available() else -1

        # Smart Selection of Attention Implementation
        # Priority: Config Override > Ministral Specific > Default Safe (SDPA)
        
        # 1. Check if user specified in config (YAML)
        if "attn_implementation" in self.config:
            target_attn_impl = self.config["attn_implementation"]
        
        # 2. Ministral Specific Default (if not in config)
        elif "Ministral" in str(self.model) or "ministral" in str(self.model).lower():
            target_attn_impl = "eager"
            
        # 3. Default Fallback (SDPA is most stable for ThreadPoolExecutor)
        else:
            target_attn_impl = "sdpa"

        try:
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.model,
                model_kwargs={
                    "dtype": torch.bfloat16,
                    "attn_implementation": target_attn_impl, 
                },
                device=device,
                trust_remote_code=True,
            )
        except Exception as e:
            # Fallback if device argument fails (e.g. conflicts with accelerate auto-map)
            if self.logger:
                self.logger.warning(f"Failed to init pipeline with device={device}, falling back to device_map. Error: {e}")
            else:
                print(f"Warning: Failed to init pipeline with device={device}, falling back to device_map. Error: {e}")
            
            device_map = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.model,
                model_kwargs={
                    "dtype": torch.bfloat16,
                    "attn_implementation": target_attn_impl,
                },
                device_map=device_map,
                trust_remote_code=True,
            )
            
        # Log the actual attention implementation being used
        try:
            # Try to get attn_implementation, fallback to private attribute or default
            attn_impl = getattr(self.pipeline.model.config, "attn_implementation", None)
            if attn_impl is None:
                 attn_impl = getattr(self.pipeline.model.config, "_attn_implementation", "unknown (not in config)")

            print(f"DEBUG: Model {self.model} attention implementation: {attn_impl}")
            if self.logger:
                self.logger.info(f"Model {self.model} loaded on device: {self.pipeline.model.device}. Attn Impl: {attn_impl}")
        except Exception as e:
             # Handle any unexpected error during property access
             msg = f"Model {self.model} loaded. Attention implementation lookup failed: {e}"
             print(f"DEBUG: {msg}")
             if self.logger:
                 self.logger.info(msg)

        print(f"DEBUG: Model {self.model} loaded on device: {self.pipeline.model.device}. Requested device_id: {device_id}")

        # Ensure pad_token is set to suppress warnings and ensure correct behavior for open-end generation
        if self.pipeline.tokenizer.pad_token_id is None:
            self.pipeline.tokenizer.pad_token_id = self.pipeline.tokenizer.eos_token_id
            if self.pipeline.tokenizer.padding_side != 'left':
                self.pipeline.tokenizer.padding_side = 'left'

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

    def compute_token_count(self, messages: List[Dict]) -> int:
        try:
            prompt_str = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False)
            tokenized_ids = self.pipeline.tokenizer(prompt_str, return_tensors='pt')['input_ids']
            return tokenized_ids.shape[1]
        except Exception as e:
            # Handle Gemma/Other models that don't support system role
            if "System role not supported" in str(e):
                new_messages = self._merge_system_role(messages)
                prompt_str = self.pipeline.tokenizer.apply_chat_template(new_messages, tokenize=False)
                tokenized_ids = self.pipeline.tokenizer(prompt_str, return_tensors='pt')['input_ids']
                return tokenized_ids.shape[1]
            raise e

    def chat_once(self, continue_prefix: str = None, max_new_tokens: int = None) -> str:
        with self.lock:
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
                "max_new_tokens":  max_new_tokens if max_new_tokens is not None else self.max_tokens,
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

            # Add repetition penalty to prevent loops (Crucial for Llama-3)
            # Use config value if present, otherwise default to 1.0
            gen_kwargs["repetition_penalty"] = self.config.get("repetition_penalty", 1.0)

            # Add stop condition to prevent long generation and ensure single step logic
            gen_kwargs["stop_strings"] = ["</step>"]
            gen_kwargs["tokenizer"] = self.pipeline.tokenizer 

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

    def get_token_len(self, text: str) -> int:
        ids = self.pipeline.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        return ids.shape[1]

    def get_sequence_score(self, conversation: List[Dict], response: str) -> float:
        with self.lock:
            # Mandatory cleanup to prevent OOM during scoring
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
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

