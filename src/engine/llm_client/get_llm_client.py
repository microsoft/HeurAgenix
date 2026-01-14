import json
from typing import Dict, Union
from src.engine.llm_client.base_llm_client import BaseLLMClient


def get_llm_client(config: Dict, system_prompt: str = None, device_id: int = 0) -> BaseLLMClient:
    # Strictly enforce Dict config (from YAML)
    if not isinstance(config, dict):
        raise ValueError(f"Expected config dict, got: {type(config)}")

    llm_type = config.get("type", "TransformersClient") # Default to Transformers if missing
    
    if llm_type == "TransformersClient":
        from src.engine.llm_client.transformers_client import TransformersClient
        llm_client = TransformersClient(config=config, system_prompt=system_prompt, device_id=device_id)
    else:
        raise ValueError(f"Unknown client type: {llm_type}")
    return llm_client