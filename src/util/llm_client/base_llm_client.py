import os
import json
from time import sleep
from typing import Dict, List, Tuple


class BaseLLMClient:
    def __init__(
            self,
            config: dict,
            prompt_dir: str=None,
            output_dir: str=None,
        ):
        self.prompt_dir = prompt_dir
        self.output_dir = output_dir
        self.config = config
        
        self.name = config.get("name", "unknown_model")
        self.top_p = config.get("top-p", 0.7)
        self.temperature = config.get("temperature", 0.95)
        self.max_tokens = config.get("max_tokens", 3200)
        self.seed = config.get("seed", None)
        self.think = config.get("think", False)
        self.max_attempts = config.get("max_attempts", 50)
        self.sleep_time = config.get("sleep_time", 60)
        self.reset(output_dir)

    def reset(self, output_dir:str=None) -> None:
        self.messages = []
        if output_dir is not None:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

    def chat_once(self) -> str:
        pass

    def chat(self) -> str:
        for index in range(self.max_attempts):
            try:
                response_content = self.chat_once()
                self.messages.append({"role": "assistant", "content": [{"type": "text", "text": response_content}]})
                return response_content
            except Exception as e:
                print(f"Try to chat {index + 1} time: {e}")
                sleep_time = self.sleep_time
                sleep(sleep_time)
        self.messages.append({"role": "assistant", "content": "Exceeded the maximum number of attempts"})
        self.dump("error")
        return None

    def dump(self, output_name: str=None) -> str:
        if self.output_dir != None and output_name != None:
            json_output_file = os.path.join(self.output_dir, f"{output_name}.json")
            text_output_file = os.path.join(self.output_dir, f"{output_name}.txt")
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