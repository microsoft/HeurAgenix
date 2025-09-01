import faiss
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


def embedding_question(questions, model):
    question_embeddings = model.encode(
        questions,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    faiss.normalize_L2(question_embeddings)
    d = question_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(question_embeddings)
    return index


def retrieve_topk_faiss_batch(index, queries, embedding_model, top_k=3, batch_size=64):
    q_emb = embedding_model.encode(
        queries,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=batch_size
    )
    q_emb = q_emb.astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    return indices

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

def get_score_single(
    model: torch.nn.Module,
    tokenizer,
    holdout_dataset,
    train_dataset,
    embedding_model,
    embedding_index,
    top_k: int = 3,
    batch_size: int = 4,
):
    system_prompt = "You are a helpful assistant."
    scores = []

    n = len(train_dataset)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = [train_dataset[i] for i in range(start, end)]

        batch_questions = [ex["message"][0]["content"] for ex in batch]
        batch_answers   = [ex["message"][1]["content"] for ex in batch]

        topk_indices = retrieve_topk_faiss_batch(
            embedding_index,
            batch_questions,
            embedding_model,
            top_k=top_k,
            batch_size=max(32, batch_size)
        )

        prompts_base = []
        prompts_with_example = []

        for qi, q in enumerate(batch_questions):
            messages_base = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": q},
                {"role": "assistant", "content": ""},
            ]
            prompt_base = tokenizer.apply_chat_template(
                messages_base, add_generation_prompt=True, tokenize=False
            )
            prompts_base.append(prompt_base)

            indices = topk_indices[qi].tolist()
            example_prompt = ""
            for i in indices:
                ex_q = holdout_dataset[i]["message"][0]["content"]
                ex_a = holdout_dataset[i]["message"][1]["content"]
                example_prompt += "Prefer responses the questions follow examples:\n"
                example_prompt += f"Question: {ex_q}\n"
                example_prompt += f"Answer: {ex_a}\n"
            example_prompt += "\n\nPlease answer the following question:"

            messages_with_example = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": example_prompt + q + "\nAnswer:"},
                {"role": "assistant", "content": ""},
            ]
            prompt_with_example = tokenizer.apply_chat_template(
                messages_with_example, add_generation_prompt=True, tokenize=False
            )
            prompts_with_example.append(prompt_with_example)

        logprob_base         = calculate_logprob_batch(model, tokenizer, prompts_base, batch_answers)
        logprob_with_example = calculate_logprob_batch(model, tokenizer, prompts_with_example, batch_answers)

        batch_scores = (logprob_with_example - logprob_base).tolist()
        scores.extend(batch_scores)

    return scores

def get_weight(
    train_dataset,
    holdout_dataset,
    model: torch.nn.Module,
    tokenizer,
    top_k: int=3,
    normalization: str=None,
    batch_size: int=4,
    embedding_model_name: str="all-mpnet-base-v2",
    cache_weight_file: str=None,
    **kwargs,
) -> np.ndarray:

    embedding_model = SentenceTransformer(embedding_model_name)
    holdout_questions = [holdout_data['message'][0]["content"] for holdout_data in holdout_dataset]
    embedding_index = embedding_question(holdout_questions, embedding_model)

    scores = get_score_single(
        model=model,
        tokenizer=tokenizer,
        holdout_dataset=holdout_dataset,
        train_dataset=train_dataset,
        embedding_model=embedding_model,
        embedding_index=embedding_index,
        top_k=top_k,
        batch_size=batch_size,
    )

    np_scores = np.array(scores, dtype=np.float32)

    if normalization == "min_max":
        mn = float(np_scores.min())
        mx = float(np_scores.max())
        normed_scores = (np_scores - mn) / (mx - mn + 1e-12)
    else:
        normed_scores = np_scores

    if cache_weight_file:
        Path(cache_weight_file).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_weight_file, normed_scores)
        np.save(cache_weight_file.split('.npy')[0] + ".raw.npy", np_scores)

    return normed_scores