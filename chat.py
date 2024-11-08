import os
from src.util.gpt_helper import GPTHelper

prompt_dir = os.path.join("output", "chat")

previous_file = "previous"
message_file = "message"
output_file = "output"

gpt_helper = GPTHelper(
    prompt_dir=prompt_dir,
    output_dir=prompt_dir
)

gpt_helper.load_chat(previous_file)
gpt_helper.load(message_file)
gpt_helper.chat()
gpt_helper.dump("output")