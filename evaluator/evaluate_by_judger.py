import os
import re
import json
from time import sleep
from evaluator.generate_output import generate_standard_answer, generate_output

def extract_winner(response: str) -> int:
    json_re = re.compile(r'\{[^}]*"winner"\s*:\s*[0-9]+\s*[^}]*\}', re.DOTALL)
    match = json_re.search(response)
    if match:
        json_str = match.group(0)
        winner = json.loads(json_str).get("winner", 0)
        return winner
    return 0


def evaluate_by_judger(client, prompt_template_file: str, standard_dict: dict, answer_dict_1: dict, answer_dict_2: dict=None, output_dir: str=None):
    prompt_template = open(prompt_template_file).read()
    assert len(standard_dict) == len(answer_dict_1)
    winners = [0, 0, 0]
    if answer_dict_2 is None:
        answer_dict_2 = standard_dict

    for index in range(len(standard_dict)):
        assert [index]["instruction"] == answer_dict_1[index]["instruction"]
        instruction = standard_dict[index]["instruction"]
        standard_output = standard_dict[index]["output"]
        output_1 = answer_dict_1[index]["output"]
        output_2 = answer_dict_2[index]["output"]
        prompt = prompt_template
        prompt = prompt.replace("{instruction}", instruction)
        prompt = prompt.replace("{standard_answer}", standard_output)
        prompt = prompt.replace("{output_1}", output_1)
        prompt = prompt.replace("{output_2}", output_2)

        response = client.chat(prompt)
        winner = extract_winner(response) - 1
        winners[winner] += 1
        sleep(0.1)
    results = {
        "tie": winners[0],
        "win": winners[1],
        "lose": winners[2],
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "bert_score_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def compare(
        model,
        tokenizer,
        test_dataset,
        prompt_template_file: str="evaluator/eval_prompt.txt",
        length_control: bool=False,
        output_dir: str=None,
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

    if os.path.exists(os.path.join(output_dir, "standard_answer.json")):
        with open(os.path.join(output_dir, "standard_answer.json"), "r", encoding="utf-8") as f:
            standard_answer = json.load(f)
    else:
        standard_answer = generate_standard_answer(test_dataset, os.path.join(output_dir, "standard_answer.json"))

    if os.path.exists(os.path.join(output_dir, "output.json")):
        with open(os.path.join(output_dir, "output.json"), "r", encoding="utf-8") as f:
            test_output = json.load(f)
    else:
        test_output = generate_output(model, tokenizer, test_dataset, 256, 4, os.path.join(output_dir, "output.json"))

    winners = evaluate_by_judger(client, prompt_template_file, standard_answer, standard_answer, test_output, length_control, output_dir)
    return winners
