import os
import json
import torch
from tqdm import tqdm


def generate_standard_answer(test_dataset, output_dir: str=None) -> list:
    try:
        standard_answer = [{"instruction": data["message"][-2]["content"], "output": data["message"][-1]["content"]} for data in test_dataset]
    except:
        standard_answer = [{"instruction": data["chosen_message"][-2]["content"], "output": data["chosen_message"][-1]["content"]} for data in test_dataset]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "standard_answer.json"), "w", encoding="utf-8") as f:
            json.dump(standard_answer, f, ensure_ascii=False, indent=2)
    return standard_answer


def generate_output(
        model,
        tokenizer,
        test_dataset,
        max_new_tokens: int=256,
        batch_size: int=4,
        enable_thinking: bool=False,
        output_dir: str = None,
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

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "output.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    tokenizer.padding_side = prev_side
    return results