import os
import json
from typing import Dict, List, Tuple
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from src.util.llm_client.base_llm_client import BaseLLMClient


class AzureGPTClient(BaseLLMClient):
    def __init__(
            self,
            config: dict,
            prompt_dir: str=None,
            output_dir: str=None,
        ):
        super().__init__(config, prompt_dir, output_dir)

        self.api_version = config["api_version"]
        self.model = config["model"]
        self.azure_endpoint = config["azure_endpoint"]

        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            azure_ad_token_provider=token_provider,
            api_version=self.api_version,
            max_retries=5,
        )


    def reset(self, output_dir:str=None) -> None:
        self.messages = []
        if output_dir is not None:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

    def chat_once(self) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            seed=self.seed,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False,
        )
        response_content = response.choices[-1].message.content
        return response_content

    def chat_once_with_tools(self, tools: List[Dict] = None) -> Tuple[str, List[Tuple[str, Dict]]]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=tools,
            tool_choice="auto",
            seed=self.seed,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False,
        )

        function_name_parameters = []
        response_content = str(response.choices[-1].message.content)
        for tool_call in response.choices[-1].message.tool_calls:
            function_name = tool_call.function.name
            parameters = json.loads(tool_call.function.arguments)
            function_name_parameters.append((function_name, parameters))
        return response_content, function_name_parameters
