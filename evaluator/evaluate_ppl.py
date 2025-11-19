import os
import json
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm


def compute_ppl(
    model,
    tokenizer,
    test_dataset,
    batch_size: int = 4,
    enable_thinking: bool = False,
    output_dir: str = None,
    **kwargs
):
    model.eval()
    device = next(model.parameters()).device

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    total_loss_sum = 0.0
    total_tok_cnt = 0

    if 'message' in test_dataset[0].keys():
        system_prompts = [data["message"][0]["content"] for data in test_dataset]
        questions      = [data["message"][1]["content"] for data in test_dataset]
        refs           = [data["message"][2]["content"] for data in test_dataset]
    elif 'chosen_message' in test_dataset[0].keys():
        system_prompts = [data["chosen_message"][0]["content"] for data in test_dataset]
        questions      = [data["chosen_message"][1]["content"] for data in test_dataset]
        refs           = [data["chosen_message"][2]["content"] for data in test_dataset]

    with torch.no_grad():
        for i in tqdm(range(0, len(questions), batch_size)):
            sys_b = system_prompts[i:i+batch_size]
            q_b   = questions[i:i+batch_size]
            ref_b = refs[i:i+batch_size]

            messages_list = []
            for s, q in zip(sys_b, q_b):
                messages_list.append(
                    [
                        {"role":"system","content":s},
                        {"role":"user",  "content":q},
                    ]
                )

            prompts = tokenizer.apply_chat_template(
                messages_list,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=enable_thinking
            )

            full_texts = [p + r for p, r in zip(prompts, ref_b)]

            enc_prompt = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False
            )
            prompt_lens = enc_prompt.attention_mask.sum(dim=1)

            enc_full = tokenizer(
                full_texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False
            )
            input_ids = enc_full.input_ids.to(device)
            attn_mask = enc_full.attention_mask.to(device)

            labels = input_ids.clone()
            labels[attn_mask == 0] = -100
            for bi in range(labels.size(0)):
                pl = int(prompt_lens[bi].item())
                labels[bi, :pl] = -100

            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            vocab = shift_logits.size(-1)
            loss_tok = F.cross_entropy(
                shift_logits.view(-1, vocab),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100
            ).view(shift_labels.size())

            mask = (shift_labels != -100)
            loss_sum_per_sample = (loss_tok * mask).sum(dim=1)
            tok_cnt_per_sample  = mask.sum(dim=1).clamp(min=1)

            total_loss_sum += loss_sum_per_sample.sum().item()
            total_tok_cnt  += tok_cnt_per_sample.sum().item()

    avg_nll = total_loss_sum / max(total_tok_cnt, 1)
    avg_ppl = math.exp(avg_nll)

    tokenizer.padding_side = prev_side

    results = {
        "avg_nll": avg_nll,
        "avg_ppl": avg_ppl,
        "loss_sum": total_loss_sum,
        "tok_cnt": total_tok_cnt,
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "ppl_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    return results