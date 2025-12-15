import json
import os
from src.util.llm_client.base_llm_client import BaseLLMClient


def get_llm_client(
        config_file: str,
        prompt_dir: str=os.path.join("src", "problems", "base", "prompt"),
        output_dir: str=None,
        ) -> BaseLLMClient:
    config = json.load(open(config_file))
    llm_type = config["type"]
    if llm_type == "azure_gpt":
        from src.util.llm_client.azure_gpt_client import AzureGPTClient
        llm_client = AzureGPTClient(config=config, prompt_dir=prompt_dir, output_dir=output_dir)
    elif llm_type == "api_model":
        from src.util.llm_client.api_model_client import APIModelClient
        llm_client = APIModelClient(config=config, prompt_dir=prompt_dir, output_dir=output_dir)
    elif llm_type == "local_model":
        from src.util.llm_client.local_model_client import LocalModelClient
        llm_client = LocalModelClient(config=config, prompt_dir=prompt_dir, output_dir=output_dir)
    elif llm_type == "vllm":
        from src.util.llm_client.vllm_client import VLLMClient
        llm_client = VLLMClient(config=config, prompt_dir=prompt_dir, output_dir=output_dir)
    llm_client.name = config.get("name", config_file.split(os.sep)[-1].split(".")[0])
    return llm_client