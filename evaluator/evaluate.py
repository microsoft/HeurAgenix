import json
import os
import torch
import re
import json
from time import sleep
from typing import List
from azure_gpt_client import AzureGPTClient


def generate_output(model, tokenizer, questions: List[str], max_new_tokens, batch_size: int = 4, output_file: str = "output.json") -> dict:
    model.eval()
    device = next(model.parameters()).device
    system_prompt = "You are a helpful assistant."
    results = []

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [tid for tid in [tokenizer.eos_token_id, eot_id] if tid is not None]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in range(0, len(questions), batch_size):
        questions_batch = questions[i : i + batch_size]
        messages_list = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question},
            ]
            for question in questions_batch
        ]
        prompts = tokenizer.apply_chat_template(
            messages_list,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            padding=True
        ).to(device)

        input_ids = prompts
        attention_mask = torch.ne(input_ids, tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id).to(device)
        input_lengths = attention_mask.sum(dim=1)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.eos_token_id,
            )

        for idx, question in enumerate(questions_batch):
            gen_tokens = outputs[idx, input_lengths[idx]:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            results.append({"instruction": question, "output": gen_text})

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def extract_winner(response: str) -> int:
    json_re = re.compile(r'\{[^}]*"winner"\s*:\s*[0-9]+\s*[^}]*\}', re.DOTALL)
    match = json_re.search(response)
    if match:
        json_str = match.group(0)
        winner = json.loads(json_str).get("winner", 0)
        return winner
    return 0


def evaluate(client: AzureGPTClient, prompt_template_file: str, output_1: dict, output_2: dict, length_control: str=None, output_file: str=None):
    prompt_template = open(prompt_template_file).read()
    assert len(output_1) == len(output_2)
    winners = [0, 0, 0]

    for index in range(len(output_1)):
        assert output_1[index]["instruction"] == output_2[index]["instruction"]
        instruction = output_1[index]["instruction"]
        output_1 = output_1[index]["output"]
        output_2 = output_2[index]["output"]
        if length_control == "min_length":
            length = min(len(output_1), len(output_2))
            output_1 = output_1[:length]
            output_2 = output_2[:length]
        prompt = prompt_template.replace("{instruction}", instruction).replace("{output_1}", output_1).replace("{output_2}", output_2)
        response = client.chat(prompt)
        winner = extract_winner(response)
        winners[winner] += 1
        if index % 20 == 0:
            file = open(output_file, "a")
            file.write(str(index) + ":" + ",".join([str(i) for i in winners]) + "\n")
            file.close()
        sleep(0.1)
    return winners


def main(model, tokenizer, test_dataset, output_dir, config: dict) -> float:
    length_control = config.get("length_control", False)
    prompt_template_file = config.get("prompt_template_file", "evaluator/eval_prompt.txt")
    test_output = os.path.join(output_dir, "test_output.json")
    comparison_output = os.path.join(output_dir, "comparison_output.json")

    gpt_setting = {
        "api_type": "azure",
        "api_version": "2025-01-01-preview",
        "azure_endpoint": "https://gcraoai9sw1.openai.azure.com/",
        "model": "gpt-4o_2024-08-06",
    }
    client = AzureGPTClient(gpt_setting)
    questions = []
    baseline_output = []
    for data in test_dataset:
        questions.append(data["message"][0]["content"])
        baseline_output.append([{"instruction": data["message"][0]["content"], "output": data["message"][1]["content"]} for data in test_dataset])
    test_output = generate_output(model, tokenizer, questions, 256, 4, os.path.join(output_dir, "test_output.json"))

    winners = evaluate(client, prompt_template_file, test_output, baseline_output, length_control, comparison_output)
    return winners / len(questions)