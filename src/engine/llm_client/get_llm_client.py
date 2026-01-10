import json
from src.engine.llm_client.base_llm_client import BaseLLMClient


def get_llm_client(config_file: str, system_prompt: str = None, device_id: int = 0) -> BaseLLMClient:
    config = json.load(open(config_file))
    llm_type = config["type"]
    if llm_type == "TransformersClient":
        from src.engine.llm_client.transformers_client import TransformersClient
        llm_client = TransformersClient(config=config, system_prompt=system_prompt, device_id=device_id)
    return llm_client