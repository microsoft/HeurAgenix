import os
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
        self.top_p = config.get("top-p", 0.7)
        self.temperature = config.get("temperature", 0.95)
        self.max_tokens = config.get("max_tokens", 3200)
        self.seed = config.get("seed", None)
        self.max_attempts = config.get("max_attempts", 50)
        self.sleep_time = config.get("sleep_time", 60)

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
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            seed=self.seed,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False,
        )
        response_content = response.choices[-1].message.content
        return response_content
