import os
from src.util.api_model_client import APIModelClient
from src.util.azure_gpt_client import AzureGPTClient

llm_client = APIModelClient(output_dir=os.path.join("output", "chat"))
llm_client.load("Hi. Are you awake?")
response = llm_client.chat()

print(response)