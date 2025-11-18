import torch
import torch.nn.functional as F
from tqdm import tqdm
import json, os

def compute_ppl(
    model,
    tokenizer,
    test_dataset,
    batch_size: int = 4,
    enable_thinking: bool = False,
):
    import math
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    model.eval()
    device = next(model.parameters()).device

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    total_loss_sum = 0.0
    total_tok_cnt = 0

    # For simple, assume test_dataset is a list of dict with keys: "message"
    system_prompts = [ex["message"][0]["content"] for ex in test_dataset]
    questions      = [ex["message"][1]["content"] for ex in test_dataset]
    refs           = [ex["message"][2]["content"] for ex in test_dataset]

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

    return {
        "avg_nll": avg_nll,
        "avg_ppl": avg_ppl,
        "loss_sum": total_loss_sum,
        "tok_cnt": total_tok_cnt,
    }

def compute_ppl_distributed(
    model,
    tokenizer,
    test_dataset,
    batch_size: int = 4,
    enable_thinking: bool = False,
):
    import os
    import math
    import torch
    import torch.distributed as dist

    dist_inited = dist.is_available() and dist.is_initialized()
    if dist_inited:
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world, local_rank = 0, 1, 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    total = len(test_dataset)
    shard_idx = list(range(rank, total, world))
    sub_dataset = test_dataset.select(shard_idx) if hasattr(test_dataset, "select") else [test_dataset[i] for i in shard_idx]

    part = compute_ppl(
        model=model,
        tokenizer=tokenizer,
        test_dataset=sub_dataset,
        batch_size=batch_size,
        enable_thinking=enable_thinking,
    )

    loss_sum_local = torch.tensor([part["loss_sum"]], dtype=torch.float64, device=device)
    tok_cnt_local  = torch.tensor([part["tok_cnt"]],  dtype=torch.float64, device=device)

    if dist_inited:
        dist.all_reduce(loss_sum_local, op=dist.ReduceOp.SUM)
        dist.all_reduce(tok_cnt_local,  op=dist.ReduceOp.SUM)

        loss_sum_g = loss_sum_local.item()
        tok_cnt_g  = tok_cnt_local.item()
        avg_nll = loss_sum_g / max(tok_cnt_g, 1.0)
        avg_ppl = float(math.exp(avg_nll))
        if rank == 0:
            return {"avg_nll": avg_nll, "avg_ppl": avg_ppl}
        else:
            return None
    else:
        return {"avg_nll": part["avg_nll"], "avg_ppl": part["avg_ppl"]}