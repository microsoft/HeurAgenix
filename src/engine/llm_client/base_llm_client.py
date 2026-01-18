import logging
import traceback
from time import sleep
from typing import Dict, List, Optional


class BaseLLMClient:
    def __init__(
            self,
            config: Dict,
            system_prompt: str = None,
            logger: Optional[logging.Logger] = None
        ):
        # Config is now passed directly as a Dict
        self.config = config
        self.logger = logger
        
        self.name = self.config.get("name", "unknown_model")
        # Support both hyphen and underscore for compatibility
        self.top_p = self.config.get("top_p", self.config.get("top-p", 1.0))
        self.temperature = self.config.get("temperature", 1.0)
        self.max_tokens = self.config.get("max_tokens", 32000)
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
        # Legacy method removed
        raise NotImplementedError("Config loading from file is deprecated. Pass dict directly.")

    def chat_once(self) -> str:
        pass

    def compute_token_count(self, messages: List[Dict]) -> int:
        """
        Computes the number of tokens for the given messages.
        """
        raise NotImplementedError("compute_token_count is not implemented")

    def chat(self, continue_prefix: str = None, max_new_tokens: int = None) -> str:
        for index in range(self.max_attempts):
            try:
                response_content = self.chat_once(continue_prefix=continue_prefix, max_new_tokens=max_new_tokens)
                self.messages.append({"role": "assistant", "content": response_content})
                return response_content
            except Exception as e:
                msg = f"Try to chat {index + 1} time: {e}"
                if self.logger:
                    self.logger.warning(msg)
                    self.logger.warning(traceback.format_exc())
                else:
                    traceback.print_exc()
                sleep(self.sleep_time)
        error_msg = "Exceeded the maximum number of attempts"
        if self.logger:
            self.logger.error(error_msg)

        return None

    def chat_once(self, continue_prefix: str = None, max_new_tokens: int = None) -> str:
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