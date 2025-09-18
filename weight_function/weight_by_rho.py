import torch
import numpy as np


def calculate_logprob_batch(
    model,
    tokenizer,
    prompts: list[str],
    responses: list[str],
):
    assert len(prompts) == len(responses)
    device = next(model.parameters()).device

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    input_ids_list = []
    attn_mask_list = []
    labels_list = []

    for p, r in zip(prompts, responses):
        enc_p = tokenizer(p, add_special_tokens=False)
        enc_r = tokenizer(r, add_special_tokens=False)
        p_ids = enc_p["input_ids"]
        r_ids = enc_r["input_ids"]

        input_ids = p_ids + r_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(p_ids) + list(r_ids)

        input_ids_list.append(input_ids)
        attn_mask_list.append(attention_mask)
        labels_list.append(labels)

    max_len = max(len(x) for x in input_ids_list)
    for i in range(len(input_ids_list)):
        cur_len = len(input_ids_list[i])
        pad_len = max_len - cur_len
        if pad_len > 0:
            input_ids_list[i] = input_ids_list[i] + [pad_id] * pad_len
            attn_mask_list[i] = attn_mask_list[i] + [0] * pad_len
            labels_list[i]    = labels_list[i]    + [-100] * pad_len

    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attn_mask_list, dtype=torch.long, device=device)
    labels = torch.tensor(labels_list, dtype=torch.long, device=device)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels != -100

    safe_labels = torch.where(valid, shift_labels, torch.zeros_like(shift_labels))
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * valid.float()

    per_sample_logprob = token_logp.sum(dim=1)
    return per_sample_logprob.detach().cpu().numpy()


def get_weight_sft(
    train_dataset,
    holdout_dataset,
    model: torch.nn.Module,
    tokenizer,
    top_k: int=3,
    batch_size: int=4,
    embedding_model_name: str="all-mpnet-base-v2",
    **kwargs,
) -> np.ndarray:
    model.eval()
    n = len(train_dataset)
    scores = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = [train_dataset[i] for i in range(start, end)]

        batch_system_prompts  = [ex["message"][0]["content"] for ex in batch]
        batch_questions       = [ex["message"][1]["content"] for ex in batch]
        batch_answers         = [ex["message"][2]["content"] for ex in batch]

        prompts_base = []

        for index, question in enumerate(batch_questions):
            system_prompt = batch_system_prompts[index]
            messages_base = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question},
            ]
            prompt_base = tokenizer.apply_chat_template(
                messages_base, add_generation_prompt=True, tokenize=False
            )
            prompts_base.append(prompt_base)
        try:
            logprob = calculate_logprob_batch(model, tokenizer, prompts_base, batch_answers).tolist()
            scores.extend(logprob)
        except Exception as e:
            for i in range(batch_size):
                logprob = calculate_logprob_batch(model, tokenizer, [prompts_base[i]], [batch_answers[i]]).tolist()[0]
                scores.append(logprob)
    return scores


def get_weight_dpo_gap(
    train_dataset,
    model: torch.nn.Module,
    tokenizer,
    batch_size: int = 8,
    **kwargs,
) -> np.ndarray:
    model.eval()
    scores = []

    n = len(train_dataset)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = [train_dataset[i] for i in range(start, end)]

        prompts   = [ex["prompt"]   for ex in batch]
        chosens   = [ex["chosen"]   for ex in batch]
        rejecteds = [ex["rejected"] for ex in batch]

        try:
            logprob_chosen = calculate_logprob_batch(model, tokenizer, prompts, chosens)
            logprob_rejected = calculate_logprob_batch(model, tokenizer, prompts, rejecteds)
            gap  = (np.asarray(logprob_chosen) - np.asarray(logprob_rejected)).tolist()
            scores.extend(gap)
        except Exception:
            for j in range(len(batch)):
                logprob_chosen = calculate_logprob_batch(model, tokenizer, [prompts[j]], [chosens[j]])
                logprob_rejected = calculate_logprob_batch(model, tokenizer, [prompts[j]], [rejecteds[j]])
                scores.append(float(logprob_chosen[0] - logprob_rejected[0]))
    return np.asarray(scores, dtype=np.float64)