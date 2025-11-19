import os
import re
import json
import random
import hashlib
from typing import Dict, Tuple, Optional, List

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

ANS_PAT = re.compile(r"####\s*([-]?\d+(?:\.\d+)?)")

def parse_gsm8k_answer(answer_text: str) -> Tuple[str, Optional[str]]:
    parts = answer_text.split("####")
    source_cot = parts[0].strip()
    source_answer = None
    m = ANS_PAT.search(answer_text)
    if m:
        source_answer = m.group(1).strip()
    return source_cot, source_answer

def format_answer(cot: str, final_num: Optional[str]) -> str:
    cot = (cot or "").strip()
    if final_num is None:
        return cot
    return (cot + ("\n" if cot else "") + f"#### {final_num}").strip()

def dropout_cot(cot: str) -> str:
    return ""

def shuffle_cot(cot: str) -> str:
    steps = [s.strip() for s in cot.split("\n") if s.strip()]
    random.shuffle(steps)
    return "\n".join(steps)

def perturb_cot(cot: str) -> str:
    def _replace(m):
        number_str = m.group(0).replace(",", "")
        try:
            if "." in number_str:
                number = float(number_str)
                noise_number = round((random.random() * 4 - 2) * number, 2)
            else:
                number = int(number_str)
                noise_number = int((random.random() * 4 - 2) * number)
            
        except:
            return m.group(0)
        return str(noise_number)

    NUM_PAT = re.compile(r"(?<![\d.])-?\d+(?:,\d{3})*(?:\.\d+)?")
    noisy_cot = NUM_PAT.sub(_replace, cot)
    return noisy_cot

def generate_cot(question: str, tokenizer, generate_model, temperature:float=0.9, top_p: float=0.9, max_new_tokens: int=64) -> str:
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(generate_model.device)
    with torch.no_grad():
        gen = generate_model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(gen[0], skip_special_tokens=True)
    noisy_cot = text[len(prompt):].strip() if text.startswith(prompt) else text
    return noisy_cot

def build_prompt(question: str) -> str:
    return (
        "You are a helpful math tutor. Solve the problem step by step, "
        "then give the final numeric answer in the form 'Answer: <number>'.\n\n"
        f"Problem: {question}\n"
        "Solution:"
    )

def generate_noise_data(
    source_dataset,
    noise_level: Dict[str, float],
    tokenizer,
    generate_model,
    output_path = "noise_data",
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
):

    dropout_ratio  = float(noise_level.get("dropout_ratio", 0.1))
    shuffle_ratio  = float(noise_level.get("shuffle_ratio", 0.1))
    replace_ratio  = float(noise_level.get("replace_ratio", 0.1))
    generate_ratio = float(noise_level.get("generate_ratio", 0.3))
    generate_model.eval()

    records = []


    for data in source_dataset:
        question = data["question"].strip()
        source_cot, source_ans = parse_gsm8k_answer(data["answer"])

        r = random.random()
        if r < dropout_ratio:
            noisy_cot = dropout_cot(source_cot)
        elif r < dropout_ratio + shuffle_ratio:
            noisy_cot = shuffle_cot(source_cot)
        elif r < dropout_ratio + shuffle_ratio + replace_ratio:
            noisy_cot = perturb_cot(source_cot)
        elif r < dropout_ratio + shuffle_ratio + replace_ratio + generate_ratio:
            noisy_cot = generate_cot(question, tokenizer, generate_model, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)
        else:
            noisy_cot = source_cot
        noisy_answer_combined = format_answer(noisy_cot, source_ans)

        rec = {
            "question": question,
            "answer": noisy_answer_combined,
        }
        records.append(rec)

    ds = Dataset.from_list(records)
    ds.save_to_disk(output_path)

    return output_path