
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from time import sleep

class AzureGPTClient:
    def __init__(
            self,
            gpt_setting: dict=None,
        ):
        self.api_version = gpt_setting["api_version"]
        self.model = gpt_setting["model"]
        self.azure_endpoint = gpt_setting["azure_endpoint"]
        self.temperature = gpt_setting.get("temperature", 0)
        self.top_p = gpt_setting.get("top_p", 0.95)
        self.seed = gpt_setting.get("seed", None)
        self.max_tokens = gpt_setting.get("max_tokens", 1600)
        self.max_attempts = gpt_setting.get("max_attempts", 10)
        self.sleep_time = gpt_setting.get("sleep_time", 10)
    
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            azure_ad_token_provider=token_provider,
            api_version=self.api_version,
            max_retries=5,
        )

    def chat(self, prompt: str):
        messages = [{"role":"user","content":prompt}]
        for _ in range(self.max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    seed=gpt_setting.get("seed", None),
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    stream=False,
                )
                response_content = response.choices[-1].message.content
                return response_content
            except:
                sleep(self.sleep_time)
                continue
