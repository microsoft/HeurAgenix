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

    def chat_with_logprobs(self) -> Tuple[str, List[Dict[str, float]], List[str]]:
        format_messages = self._format_messages(self.messages)

        text = self.pipeline.tokenizer.apply_chat_template(
            format_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.think,
        )
        
        inputs = self.pipeline.tokenizer(text, return_tensors="pt")
        if hasattr(self.pipeline.model, "device"):
            inputs = {k: v.to(self.pipeline.model.device) for k, v in inputs.items()}

        outputs = self.pipeline.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )
        
        generated_tokens_ids = outputs.sequences[:, inputs["input_ids"].shape[-1]:]
        response_content = self.pipeline.tokenizer.decode(generated_tokens_ids[0], skip_special_tokens=True)

        logprobs_data = []
        generated_tokens_list = []
        
        # Iterate over each generation step
        for i, score_tensor in enumerate(outputs.scores):
            # Get the token id that was actually generated at this step
            # Note: generated_tokens_ids[0][i] is the token id
            if i < len(generated_tokens_ids[0]):
                token_id = generated_tokens_ids[0][i]
                token_str = self.pipeline.tokenizer.decode([token_id.item()])
                generated_tokens_list.append(token_str)
            
            probs = torch.nn.functional.softmax(score_tensor[0], dim=-1)
            top_probs, top_indices = torch.topk(probs, k=20)
            
            step_logprobs = {}
            for prob, idx in zip(top_probs, top_indices):
                token = self.pipeline.tokenizer.decode([idx.item()])
                step_logprobs[token] = torch.log(prob).item()
            logprobs_data.append(step_logprobs)
            
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": response_content}]})
            
        return response_content, logprobs_data, generated_tokens_list

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

