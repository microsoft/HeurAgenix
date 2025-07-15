import os
from src.util.llm_client.get_llm_client import get_llm_client

config_file = os.path.join("output", "llm_config", "azure_gpt_4o.json")
llm_client = get_llm_client(config_file=config_file)
llm_client.load("Hi. Are you awake?")
response = llm_client.chat()
print(response)