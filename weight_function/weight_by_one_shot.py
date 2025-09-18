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
    top_k: int = 3,
    batch_size: int = 4,
    embedding_model_name: str = "all-mpnet-base-v2",
    **kwargs,
) -> np.ndarray:
    embedding_model, embedding_index = init_query(embedding_model_name, holdout_dataset)
    n = len(train_dataset)
    h = len(holdout_dataset)

    holdout_logprobs = []
    holdout_system_prompts = [ex["message"][0]["content"] for ex in holdout_dataset]
    holdout_questions      = [ex["message"][1]["content"] for ex in holdout_dataset]
    holdout_answers        = [ex["message"][2]["content"] for ex in holdout_dataset]
    for start in range(0, h, batch_size):
        end = min(start + batch_size, h)

        batch = [holdout_dataset[i] for i in range(start, end)]
        batch_prompts = []
        batch_answers = []
        for index in range(start, end):
            system_prompt = holdout_system_prompts[index]
            question      = holdout_questions[index]
            answer        = holdout_answers[index]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question},
                {"role": "assistant", "content": ""},
            ]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            batch_prompts.append(prompt)
            batch_answers.append(answer)
        try:
            logprob_base = calculate_logprob_batch(model, tokenizer, batch_prompts, batch_answers)
            holdout_logprobs.extend(logprob_base.tolist())
        except Exception as e:
            for i in range(batch_size):
                logprob_base = calculate_logprob_batch(model, tokenizer, [batch_prompts[i]], [batch_answers[i]])
                holdout_logprobs.append(logprob_base.tolist()[0])

    scores = []
    for index in range(0, n):

        system_prompt = train_dataset[index]["message"][0]["content"]
        question      = train_dataset[index]["message"][1]["content"]
        answer        = train_dataset[index]["message"][2]["content"]

        topk_index = retrieve_topk_faiss_batch(
            embedding_index,
            [question],
            embedding_model,
            top_k=top_k,
            batch_size=1,
        )[0]

        single_data_prompts_with_example = []
        single_data_holdout_answers = []
        single_data_holdout_logprobs = []
        for k in topk_index.tolist():
            example_prompt = ""
            example_prompt += "Prefer responses the questions follow examples:\n"
            example_prompt += f"Question: {question}\n"
            example_prompt += f"Answer: {answer}\n"
            example_prompt += "\n\nPlease answer the following question:"

            messages_with_example = [
                {"role": "system", "content": holdout_system_prompts[k]},
                {"role": "user",   "content": example_prompt + holdout_questions[k] + "\nAnswer:"},
                {"role": "assistant", "content": ""},
            ]
            prompt_with_example = tokenizer.apply_chat_template(
                messages_with_example, add_generation_prompt=True, tokenize=False
            )
            single_data_prompts_with_example.append(prompt_with_example)
            single_data_holdout_answers.append(holdout_answers[k])
            single_data_holdout_logprobs.append(holdout_logprobs[k])

        try:
            single_data_logprob_with_example = calculate_logprob_batch(model, tokenizer, single_data_prompts_with_example, single_data_holdout_answers)
            score = np.average((single_data_logprob_with_example - single_data_holdout_logprobs))
            scores.append(score)
        except Exception:
            score = 0
            for i in range(top_k):
                single_data_logprob_with_example = calculate_logprob_batch(model, tokenizer, single_data_prompts_with_example[i], single_data_holdout_answers[i])
                score += (single_data_logprob_with_example - single_data_holdout_logprobs[i])
            scores.append(score / top_k)

    scores = np.asarray(scores, dtype=np.float32)
    return scores


def get_weight_preference(
    train_dataset,
    holdout_dataset,
    model: torch.nn.Module,
    tokenizer,
    top_k: int = 3,
    batch_size: int = 4,
    embedding_model_name: str = "all-mpnet-base-v2",
    **kwargs,
) -> np.ndarray:
    embedding_model, embedding_index = init_query(embedding_model_name, holdout_dataset)
    n = len(train_dataset)
    h = len(holdout_dataset)

    holdout_chosen_logprobs   = []
    holdout_rejected_logprobs = []
    holdout_system_prompts    = [ex["chosen_message"][0]["content"] for ex in holdout_dataset]
    holdout_questions         = [ex["chosen_message"][1]["content"] for ex in holdout_dataset]
    holdout_chosen_answers    = [ex["chosen_message"][2]["content"] for ex in holdout_dataset]
    holdout_rejected_answers  = [ex["rejected_message"][2]["content"] for ex in holdout_dataset]

    for start in range(0, h, batch_size):
        end = min(start + batch_size, h)
        batch_prompts = []
        batch_chosen_answers = []
        batch_rejected_answers = []
        for index in range(start, end):
            system_prompt = holdout_system_prompts[index]
            question      = holdout_questions[index]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question},
                {"role": "assistant", "content": ""},
            ]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            batch_prompts.append(prompt)
            batch_chosen_answers.append(holdout_chosen_answers[index])
            batch_rejected_answers.append(holdout_rejected_answers[index])
        try:
            chosen_logprob_base   = calculate_logprob_batch(model, tokenizer, batch_prompts, batch_chosen_answers)
            rejected_logprob_base = calculate_logprob_batch(model, tokenizer, batch_prompts, batch_rejected_answers)
            chosen_logprob_base   = chosen_logprob_base.detach().cpu().numpy() if hasattr(chosen_logprob_base, "detach") else np.asarray(chosen_logprob_base)
            rejected_logprob_base = rejected_logprob_base.detach().cpu().numpy() if hasattr(rejected_logprob_base, "detach") else np.asarray(rejected_logprob_base)
            holdout_chosen_logprobs.extend(chosen_logprob_base.tolist())
            holdout_rejected_logprobs.extend(rejected_logprob_base.tolist())
        except Exception:
            for i in range(len(batch_prompts)):
                c = calculate_logprob_batch(model, tokenizer, [batch_prompts[i]], [batch_chosen_answers[i]])
                r = calculate_logprob_batch(model, tokenizer, [batch_prompts[i]], [batch_rejected_answers[i]])
                c = c.detach().cpu().numpy() if hasattr(c, "detach") else np.asarray(c)
                r = r.detach().cpu().numpy() if hasattr(r, "detach") else np.asarray(r)
                holdout_chosen_logprobs.append(float(c[0]))
                holdout_rejected_logprobs.append(float(r[0]))

    holdout_chosen_logprobs   = np.asarray(holdout_chosen_logprobs, dtype=np.float32)
    holdout_rejected_logprobs = np.asarray(holdout_rejected_logprobs, dtype=np.float32)

    scores = []
    for index in range(n):
        system_prompt   = train_dataset[index]["chosen_message"][0]["content"]
        question        = train_dataset[index]["chosen_message"][1]["content"]
        chosen_answer   = train_dataset[index]["chosen_message"][2]["content"]

        topk_index = retrieve_topk_faiss_batch(
            embedding_index,
            [question],
            embedding_model,
            top_k=top_k,
            batch_size=1,
        )[0]

        single_data_prompts_with_example = []
        single_data_holdout_chosen_answers = []
        single_data_holdout_rejected_answers = []
        base_gap = []
        for k in topk_index.tolist():
            example_prompt = ""
            example_prompt += "Prefer responses the questions follow examples:\n"
            example_prompt += f"Question: {question}\n"
            example_prompt += f"Answer: {chosen_answer}\n"
            example_prompt += "\n\nPlease answer the following question:"

            messages_with_example = [
                {"role": "system", "content": holdout_system_prompts[k]},
                {"role": "user",   "content": example_prompt + holdout_questions[k] + "\nAnswer:"},
                {"role": "assistant", "content": ""},
            ]
            prompt_with_example = tokenizer.apply_chat_template(
                messages_with_example, add_generation_prompt=True, tokenize=False
            )
            single_data_prompts_with_example.append(prompt_with_example)
            single_data_holdout_chosen_answers.append(holdout_chosen_answers[k])
            single_data_holdout_rejected_answers.append(holdout_rejected_answers[k])
            base_gap.append(holdout_chosen_logprobs[k] - holdout_rejected_logprobs[k])

        try:
            with_chosen = calculate_logprob_batch(model, tokenizer, single_data_prompts_with_example, single_data_holdout_chosen_answers)
            with_rejected = calculate_logprob_batch(model, tokenizer, single_data_prompts_with_example, single_data_holdout_rejected_answers)
            with_chosen = with_chosen.detach().cpu().numpy() if hasattr(with_chosen, "detach") else np.asarray(with_chosen)
            with_rejected = with_rejected.detach().cpu().numpy() if hasattr(with_rejected, "detach") else np.asarray(with_rejected)
            diffs = (with_chosen - with_rejected) - np.asarray(base_gap, dtype=np.float32)
            score = float(diffs.mean()) if len(diffs) > 0 else 0.0
            scores.append(score)
        except Exception:
            vals = []
            for i in range(len(single_data_prompts_with_example)):
                c = calculate_logprob_batch(model, tokenizer, [single_data_prompts_with_example[i]], [single_data_holdout_chosen_answers[i]])
                r = calculate_logprob_batch(model, tokenizer, [single_data_prompts_with_example[i]], [single_data_holdout_rejected_answers[i]])
                c = c.detach().cpu().numpy() if hasattr(c, "detach") else np.asarray(c)
                r = r.detach().cpu().numpy() if hasattr(r, "detach") else np.asarray(r)
                vals.append(float((c[0] - r[0]) - base_gap[i]))
            score = float(np.mean(vals)) if len(vals) > 0 else 0.0
            scores.append(score)

    scores = np.asarray(scores, dtype=np.float32)
    return scores