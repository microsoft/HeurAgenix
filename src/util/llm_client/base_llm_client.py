import os
import json
from time import sleep
from typing import Dict, List, Tuple


class BaseLLMClient:
    def __init__(
            self,
            config_path: str,
            system_prompt: str = None
        ):
        self.config = self.load_config(config_path)
        
        
        self.name = self.config.get("name", "unknown_model")
        self.top_p = self.config.get("top-p", 0.7)
        self.temperature = self.config.get("temperature", 0.95)
        self.max_tokens = self.config.get("max_tokens", 3200)
        self.seed = self.config.get("seed", None)
        self.think = self.config.get("think", False)
        self.max_attempts = self.config.get("max_attempts", 50)
        self.sleep_time = self.config.get("sleep_time", 60)
        if system_prompt:
            self.system_prompt = system_prompt
            self.messages = [{"role": "system", "content": system_prompt}]
        else:
            self.messages = []
            self.system_prompt = None

    def load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return json.load(f)

    def chat_once(self) -> str:
        pass

    def chat(self) -> str:
        for index in range(self.max_attempts):
            try:
                response_content = self.chat_once()
                self.messages.append({"role": "assistant", "content": response_content})
                return response_content
            except Exception as e:
                print(f"Try to chat {index + 1} time: {e}")
                sleep_time = self.sleep_time
                sleep(sleep_time)
        self.messages.append({"role": "assistant", "content": "Exceeded the maximum number of attempts"})
        self.dump("error")
        return None

    def dump(self, output_path: str=None) -> str:
        json_output_file = output_path.replace(".txt", ".json")
        text_output_file = output_path.replace(".json", ".txt")
        print(f"Chat dumped to {text_output_file}")
        with open(json_output_file, "w") as fp:
            json.dump(self.messages, fp, indent=4)

        with open(text_output_file, "w", encoding="UTF-8") as file:
            for message in self.messages:
                file.write(message["role"] + "\n")
                contents = ""
                for i, content in enumerate(message["content"]):
                    if isinstance(content, dict) and content.get("type") == "text":
                        contents += content["text"]
                    elif isinstance(content, str):
                        contents += content
                file.write(contents + "\n------------------------------------------------------------------------------------\n\n")
        return self.messages[-1]["content"][0]["text"]

    def chat_once(self) -> str:
        raise NotImplemented

    def get_sequence_score(self, conversation: List[Dict], response: str) -> float:
        """
        Calculate the average NLL of the response given the conversation context.
        """
        raise NotImplementedError("get_sequence_score is not implemented for this client.")

    def reset(self, system_prompt: str = None) -> None:
        """Clears history and optionally sets a new system prompt."""
        self.messages = []
        self.system_prompt = system_prompt
        if self.system_prompt:
            self.messages = [{"role": "system", "content": self.system_prompt}]

    def add_message(self, content, role: str = "user") -> None:
        """Appends a single message to history."""
        self.messages.append({"role": role, "content": content})

    def set_history(self, messages: List[Dict]) -> None:
        """Replaces current history with provided messages."""
        # 1. Try to find system prompt in the new messages
        sys_prompt = next((m["content"] for m in messages if m["role"] == "system"), None)
        
        # 2. Reset with that prompt
        self.reset(sys_prompt)
        
        # 3. Append non-system messages
        for m in messages:
            if m["role"] != "system":
                self.add_message(m["content"], m["role"])