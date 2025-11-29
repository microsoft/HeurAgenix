import os
from src.util.llm_client.get_llm_client import get_llm_client
import sys

config_file = os.path.join("data", "llm_config", "azure_gpt_5.json")
llm_client = get_llm_client(config_file=config_file, prompt_dir=os.path.join("output", "chat"), output_dir=os.path.join("output", "chat"))
llm_client.load_chat("previous.json")
llm_client.load("message.txt")
response = llm_client.chat()
llm_client.dump("output")