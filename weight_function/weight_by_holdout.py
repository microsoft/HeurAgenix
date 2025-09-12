import os
import faiss
import torch
import numpy as np
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

def init_query(embedding_model_name, holdout_dataset):
    embedding_model = SentenceTransformer(embedding_model_name)
    try:
        holdout_questions = [holdout_data['message'][1]["content"] for holdout_data in holdout_dataset]
    except:
        holdout_questions = [holdout_data['chosen_message'][1]["content"] for holdout_data in holdout_dataset]
    embedding_index = embedding_question(holdout_questions, embedding_model)
    return embedding_model, embedding_index


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
    embedding_model, embedding_index = init_query(embedding_model_name, holdout_dataset)
    n = len(train_dataset)
    scores = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = [train_dataset[i] for i in range(start, end)]

        batch_system_prompts  = [ex["message"][0]["content"] for ex in batch]
        batch_questions       = [ex["message"][1]["content"] for ex in batch]
        batch_answers         = [ex["message"][2]["content"] for ex in batch]

        topk_indices = retrieve_topk_faiss_batch(
            embedding_index,
            batch_questions,
            embedding_model,
            top_k=top_k,
            batch_size=max(32, batch_size)
        )

        prompts_base = []
        prompts_with_example = []

        for index, question in enumerate(batch_questions):
            system_prompt = batch_system_prompts[index]
            messages_base = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question},
                {"role": "assistant", "content": ""},
            ]
            prompt_base = tokenizer.apply_chat_template(
                messages_base, add_generation_prompt=True, tokenize=False
            )
            prompts_base.append(prompt_base)

            indices = topk_indices[index].tolist()
            example_prompt = ""
            for i in indices:
                example_question = holdout_dataset[i]["message"][-2]["content"]
                example_answer = holdout_dataset[i]["message"][-1]["content"]
                example_prompt += "Prefer responses the questions follow examples:\n"
                example_prompt += f"Question: {example_question}\n"
                example_prompt += f"Answer: {example_answer}\n"
            example_prompt += "\n\nPlease answer the following question:"

            messages_with_example = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": example_prompt + question + "\nAnswer:"},
                {"role": "assistant", "content": ""},
            ]
            prompt_with_example = tokenizer.apply_chat_template(
                messages_with_example, add_generation_prompt=True, tokenize=False
            )
            prompts_with_example.append(prompt_with_example)
        try:
            logprob_base         = calculate_logprob_batch(model, tokenizer, prompts_base, batch_answers)
            logprob_with_example = calculate_logprob_batch(model, tokenizer, prompts_with_example, batch_answers)

            batch_scores = (logprob_with_example - logprob_base).tolist()
            scores.extend(batch_scores)
            print(len(scores), len(train_dataset))
        except Exception as e:
            for i in range(batch_size):
                    logprob_base         = calculate_logprob_batch(model, tokenizer, [prompts_base[i]], [batch_answers[i]])
                    logprob_with_example = calculate_logprob_batch(model, tokenizer, [prompts_with_example[i]], [batch_answers[i]])
                    score = (logprob_with_example - logprob_base).tolist()[0]
                    scores.append(score)
                    print(len(scores), len(train_dataset))

    return scores


def get_weight_preference(
    train_dataset,
    holdout_dataset,
    model: torch.nn.Module,
    tokenizer,
    top_k: int=3,
    batch_size: int=4,
    embedding_model_name: str="all-mpnet-base-v2",
    **kwargs,
) -> np.ndarray:
    embedding_model, embedding_index = init_query(embedding_model_name, holdout_dataset)
    n = len(train_dataset)
    scores = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = [train_dataset[i] for i in range(start, end)]

        batch_system_prompts   = [ex["chosen_message"][0]["content"] for ex in batch]
        batch_questions        = [ex["chosen_message"][1]["content"] for ex in batch]
        batch_chosen_answers   = [ex["chosen_message"][2]["content"] for ex in batch]
        batch_rejected_answers = [ex["rejected_message"][2]["content"] for ex in batch]

        topk_indices = retrieve_topk_faiss_batch(
            embedding_index,
            batch_questions,
            embedding_model,
            top_k=top_k,
            batch_size=max(32, batch_size)
        )

        prompts_base = []
        prompts_with_example = []

        for index, question in enumerate(batch_questions):
            system_prompt = batch_system_prompts[index]
            messages_base = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question},
                {"role": "assistant", "content": ""},
            ]
            prompt_base = tokenizer.apply_chat_template(
                messages_base, add_generation_prompt=True, tokenize=False
            )
            prompts_base.append(prompt_base)

            indices = topk_indices[index].tolist()
            example_prompt = ""
            for i in indices:
                example_question = holdout_dataset[i]["chosen_message"][-2]["content"]
                example_answer = holdout_dataset[i]["chosen_message"][-1]["content"]
                example_prompt += "Prefer responses the questions follow examples:\n"
                example_prompt += f"Question: {example_question}\n"
                example_prompt += f"Answer: {example_answer}\n"
            example_prompt += "\n\nPlease answer the following question:"

            messages_with_example = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": example_prompt + question + "\nAnswer:"},
                {"role": "assistant", "content": ""},
            ]
            prompt_with_example = tokenizer.apply_chat_template(
                messages_with_example, add_generation_prompt=True, tokenize=False
            )
            prompts_with_example.append(prompt_with_example)
        try:
            logprob_base_chosen           = calculate_logprob_batch(model, tokenizer, prompts_base, batch_chosen_answers)
            logprob_base_rejected         = calculate_logprob_batch(model, tokenizer, prompts_base, batch_rejected_answers)
            logprob_with_example_chosen   = calculate_logprob_batch(model, tokenizer, prompts_with_example,  batch_chosen_answers)
            logprob_with_example_rejected = calculate_logprob_batch(model, tokenizer, prompts_with_example,  batch_rejected_answers)
            batch_scores = ((logprob_with_example_chosen - logprob_with_example_rejected) - (logprob_base_chosen - logprob_base_rejected)).tolist()
            scores.extend(batch_scores)
            print(len(scores), len(train_dataset))
        except Exception as e:
            for i in range(batch_size):
                    logprob_base_chosen           = calculate_logprob_batch(model, tokenizer, [prompts_base[i]], [batch_chosen_answers[i]])
                    logprob_base_rejected         = calculate_logprob_batch(model, tokenizer, [prompts_base[i]], [batch_rejected_answers[i]])
                    logprob_with_example_chosen   = calculate_logprob_batch(model, tokenizer, [prompts_with_example[i]], [batch_chosen_answers[i]])
                    logprob_with_example_rejected = calculate_logprob_batch(model, tokenizer, [prompts_with_example[i]], [batch_rejected_answers[i]])

                    batch_scores = ((logprob_with_example_chosen - logprob_with_example_rejected) - (logprob_base_chosen - logprob_base_rejected)).tolist()
                    scores.extend(batch_scores)
                    print(len(scores), len(train_dataset))
    return scores