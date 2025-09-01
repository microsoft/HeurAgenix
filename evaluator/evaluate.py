import json
import os
from pathlib import Path
import torch
import re
import json
from time import sleep


def extract_winner(response: str) -> int:
    json_re = re.compile(r'\{[^}]*"winner"\s*:\s*[0-9]+\s*[^}]*\}', re.DOTALL)
    match = json_re.search(response)
    if match:
        json_str = match.group(0)
        winner = json.loads(json_str).get("winner", 0)
        return winner
    return 0


def evaluate(client, prompt_template_file: str, output_dict_1: dict, output_dict_2: dict, length_control: str=None):
    prompt_template = open(prompt_template_file).read()
    assert len(output_dict_1) == len(output_dict_2)
    winners = [0, 0, 0]

    for index in range(len(output_dict_1)):
        assert output_dict_1[index]["instruction"] == output_dict_2[index]["instruction"]
        instruction = output_dict_1[index]["instruction"]
        output_1 = output_dict_1[index]["output"]
        output_2 = output_dict_2[index]["output"]
        if length_control == "min_length":
            length = min(len(output_1), len(output_2))
            output_1 = output_1[:length]
            output_2 = output_2[:length]
        prompt = prompt_template.replace("{instruction}", instruction).replace("{output_1}", output_1).replace("{output_2}", output_2)
        response = client.chat(prompt)
        winner = extract_winner(response)
        winners[winner] += 1
        sleep(0.1)
    return winners


def generate_baseline(test_dataset, output_dir: str="output") -> dict:
    baseline_output = []
    baseline_output.append([{"instruction": data["message"][0]["content"], "output": data["message"][1]["content"]} for data in test_dataset])
    output_file = os.path.join(output_dir, "baselines.json")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(baseline_output, f, ensure_ascii=False, indent=2)


def generate_output(
        model,
        tokenizer,
        test_dataset,
        max_new_tokens: int=256,
        batch_size: int=4,
        output_dir: str="output",
        return_type: str="output_file",
        **kwargs
) -> dict:
    model.eval()
    
    model_to_use = getattr(model, "module", model)
    gen_cfg = getattr(model_to_use, "generation_config", None)

    eos_ids = None
    if gen_cfg is not None and getattr(gen_cfg, "eos_token_id", None) is not None:
        eos_ids = gen_cfg.eos_token_id
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]

    device = next(model.parameters()).device
    system_prompt = "You are a helpful assistant."
    results = []

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    questions = [data["message"][0]["content"] for data in test_dataset]
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
            tokenize=False
        )
        encode_prompts = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        input_ids = encode_prompts.input_ids.to(device)
        attention_mask = encode_prompts.attention_mask.to(device)
        input_lengths = attention_mask.sum(dim=1)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
            )

        for idx, question in enumerate(questions_batch):
            gen_tokens = outputs[idx, input_lengths[idx]:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            results.append({"instruction": question, "output": gen_text})

    output_file = os.path.join(output_dir, "test_results.json")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if return_type == "output_file":
        return output_file
    elif return_type == "output_result":
        return results


def compare(
        model,
        tokenizer,
        test_dataset,
        output_dir: str="output",
        prompt_template_file: str="evaluator/eval_prompt.txt",
        length_control: bool=False,
        **kwargs,
):
    from evaluator.azure_gpt_client import AzureGPTClient
    gpt_setting = {
        "api_type": "azure",
        "api_version": "2025-01-01-preview",
        "azure_endpoint": "https://gcraoai9sw1.openai.azure.com/",
        "model": "gpt-4o_2024-08-06",
    }
    client = AzureGPTClient(gpt_setting)

    baseline_output = generate_baseline(test_dataset, output_dir)
    test_output = generate_output(model, tokenizer, test_dataset, 256, 4, output_dir, "output_result")

    winners = evaluate(client, prompt_template_file, baseline_output, test_output, length_control)
    return winners