import os
import re
import json
import torch
import torch.distributed as dist
from time import sleep
from tqdm import tqdm

def extract_winner(response: str) -> int:
    json_re = re.compile(r'\{[^}]*"winner"\s*:\s*[0-9]+\s*[^}]*\}', re.DOTALL)
    match = json_re.search(response)
    if match:
        json_str = match.group(0)
        winner = json.loads(json_str).get("winner", 0)
        return winner
    return 0


def evaluate(client, prompt_template_file: str, baseline_dict: dict, test_dict: dict, length_control: str=None, output_file: str=None):
    prompt_template = open(prompt_template_file).read()
    assert len(baseline_dict) == len(test_dict)
    winners = [0, 0, 0]

    for index in range(len(baseline_dict)):
        assert baseline_dict[index]["instruction"] == test_dict[index]["instruction"]
        instruction = baseline_dict[index]["instruction"]
        output_1 = baseline_dict[index]["output"]
        output_2 = test_dict[index]["output"]
        if length_control == "min_length":
            length = min(len(output_1), len(output_2))
            output_1 = output_1[:length]
            output_2 = output_2[:length]
        prompt = prompt_template.replace("{instruction}", instruction).replace("{output_1}", output_1).replace("{output_2}", output_2)
        response = client.chat(prompt)
        winner = extract_winner(response)
        winners[winner] += 1
        tie, loss, win = winners
        win_rate = (win + 0.5 * tie) / (win + tie + loss)
        print(f"Win/Tie/Lose: {win}, {tie}, {loss}, win rate: {win_rate}\n")
        sleep(0.1)
    output_file = open(output_file, "w")
    tie, loss, win = winners
    output_file.write(f"Win/Tie/Lose: {win}, {tie}, {loss}, win rate: {win_rate}\n")
    output_file.close()
    return winners


def generate_baseline(test_dataset, output_file: str=None) -> list:
    baseline_output = [{"instruction": data["message"][-2]["content"], "output": data["message"][-1]["content"]} for data in test_dataset]
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(baseline_output, f, ensure_ascii=False, indent=2)
    return baseline_output


def generate_output(
        model,
        tokenizer,
        test_dataset,
        max_new_tokens: int=256,
        batch_size: int=4,
        output_file: str=None,
        enable_thinking: bool=False,
        **kwargs
) -> list:
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    device = next(model.parameters()).device
    results = []

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = []
    if getattr(model, "generation_config", None) and getattr(model.generation_config, "eos_token_id", None) is not None:
        if isinstance(model.generation_config.eos_token_id, int):
            eos_ids.append(model.generation_config.eos_token_id)
        else:
            eos_ids.extend(model.generation_config.eos_token_id)
    eos_ids.extend([tokenizer.eos_token_id, eot_id])
    eos_ids = [t for t in set(eos_ids) if t is not None]

    if 'message' in test_dataset[0].keys():
        system_prompts = [data["message"][0]["content"] for data in test_dataset]
        questions      = [data["message"][1]["content"] for data in test_dataset]
    elif 'chosen_message' in test_dataset[0].keys():
        system_prompts = [data["chosen_message"][0]["content"] for data in test_dataset]
        questions      = [data["chosen_message"][1]["content"] for data in test_dataset]

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    for i in tqdm(range(0, len(questions), batch_size)):
        messages_list = []
        system_prompts_batch = system_prompts[i : i + batch_size]
        questions_batch      = questions[i : i + batch_size]
        for index, question in enumerate(questions_batch):
            system_prompt = system_prompts_batch[index]
            messages_list.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": question},
                ]
            )
        prompts = tokenizer.apply_chat_template(
            messages_list,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking
        )
        encode_prompts = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        input_ids = encode_prompts.input_ids.to(device)
        attention_mask = encode_prompts.attention_mask.to(device)
        init_prompt_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
            )

        for idx, q in enumerate(questions_batch):
            gen_tokens = outputs[idx, init_prompt_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            results.append({"instruction": q, "output": gen_text})

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    tokenizer.padding_side = prev_side
    return results


def generate_output_distributed(
    model,
    tokenizer,
    test_dataset,
    max_new_tokens=256,
    batch_size=4,
    enable_thinking: bool=False,
    output_file=None,
    **kwargs
):
    dist_inited = dist.is_available() and dist.is_initialized()
    if dist_inited:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world_size, local_rank = 0, 1, 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) if device.type == "cuda" else None
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    total = len(test_dataset)
    global_indices = list(range(total))
    shard_indices = global_indices[rank::world_size]
    sub_dataset = test_dataset.select(shard_indices)

    partial = generate_output(
        model=model,
        tokenizer=tokenizer,
        test_dataset=sub_dataset,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        enable_thinking=enable_thinking,
        output_file=None
    )

    results_local = []
    for j, res in enumerate(partial):
        item = dict(res)
        item["index"] = shard_indices[j]
        results_local.append(item)

    if dist_inited:
        obj_list = [None for _ in range(world_size)]
        dist.all_gather_object(obj_list, results_local)
        merged = [x for sub in obj_list for x in sub] if rank == 0 else None
    else:
        merged = results_local

    if (not dist_inited) or rank == 0:
        merged.sort(key=lambda x: x["index"])
        final = [{"instruction": r["instruction"], "output": r["output"]} for r in merged]
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final, f, ensure_ascii=False, indent=2)
        return output_file
    else:
        return None


def compare(
        model,
        tokenizer,
        test_dataset,
        prompt_template_file: str="evaluator/eval_prompt.txt",
        output_file: str=None,
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
    test_dir = os.path.dirname(os.path.normpath(output_file))

    baseline_output = generate_baseline(test_dataset, os.path.join(test_dir, "baseline.json"))
    test_output = generate_output(model, tokenizer, test_dataset, 256, 4, os.path.join(test_dir, "output.json"))

    winners = evaluate(client, prompt_template_file, baseline_output, test_output, length_control, output_file)
    return winners

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dataset_loader.mix_alpaca import get_dataset

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    test_dataset = get_dataset(tokenizer)["test"]
    output_file = "output/llama-3-8b-instruct/test_results.json" if rank == 0 else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": local_rank},
        attn_implementation="flash_attention_2",
    )
    model.eval()
    model.config.use_cache = True

    generate_output_distributed(
        test_dataset=test_dataset,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        batch_size=8,
        output_file=output_file,
    )

    dist.barrier()
    if rank == 0:
        print("Done.")
