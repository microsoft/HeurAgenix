import os
from src.util.gpt_helper import GPTHelper


gpt_helper = GPTHelper(output_dir=os.path.join("output", "chat"))

gpt_helper.load("Hi GPT, are you awake?")
gpt_helper.chat()
response = gpt_helper.dump()
print(response)