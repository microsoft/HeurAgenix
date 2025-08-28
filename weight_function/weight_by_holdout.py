import faiss
import torch
import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
import torch.distributed as dist
from torch.distributed import ReduceOp
import datetime


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

def retrieve_topk_faiss(index, query, model, top_k=3):
    q_emb = model.encode(query, convert_to_tensor=False)
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)
    q_emb = q_emb.reshape(1, -1).astype("float32")

    scores, indices = index.search(q_emb, top_k)
    return indices

def calculate_logprob(model, tokenizer, prompt: str, response: str, ) -> float:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    response_inputs = tokenizer(response, return_tensors="pt").to(device)
    input_ids = torch.cat([inputs.input_ids, response_inputs.input_ids], dim=1).long()
    attention_mask = torch.cat([inputs.attention_mask, response_inputs.attention_mask], dim=1)

    labels = input_ids.clone()
    labels[:, : inputs.input_ids.size(1)] = -100
    labels = labels.long()

    with torch.no_grad():
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels)
        n_answer_tokens = (labels != -100).sum().item()
        average_logprob = - out.loss.item()
        logprob = average_logprob * n_answer_tokens
    return logprob

def evaluation_data(model, tokenizer, example_messages: list[list], target_message: list):
    target_question, target_answer = target_message[0]['content'], target_message[1]['content']
    system_prompt = "You are a helpful assistant."
    messages_base = [{"role": "system", "content": system_prompt}, {"role": "user", "content": target_question}, {"role": "assistant", "content": ""}]

    example_prompt = ""
    for example_question, example_answer in example_messages:
        example_prompt += "Prefer responses the questions follow examples:\n"
        example_prompt += f"Question: {example_question['content']}\n"
        example_prompt += f"Answer: {example_answer['content']}\n"
    example_prompt += "\n\nPlease answer the following question:"
    messages_with_example = [{"role": "system", "content": system_prompt}, {"role": "user", "content": example_prompt + target_question + "\nAnswer:"}, {"role": "assistant", "content": ""}]

    prompt_base = tokenizer.apply_chat_template(messages_base, add_generation_prompt=True, tokenize=False)
    prompt_with_example = tokenizer.apply_chat_template(messages_with_example, add_generation_prompt=True, tokenize=False)

    logprob_base         = calculate_logprob(model, tokenizer, prompt_base, target_answer)
    logprob_with_example = calculate_logprob(model, tokenizer, prompt_with_example, target_answer)
    score = logprob_with_example - logprob_base
    return score

def calculate_score(holdout_dataset, target_message, embedding_index, embedding_model, model, tokenizer):
    target_question = target_message[0]['content']
    indices = retrieve_topk_faiss(embedding_index, target_question, embedding_model, 3).flatten().tolist()
    example_messages = [holdout_dataset[i] for i in indices]
    score = evaluation_data(model, tokenizer, example_messages, target_message)

    return score


def get_score(
    model: torch.nn.Module,
    tokenizer,
    holdout_dataset,
    train_dataset,
    config: dict
) -> np.ndarray:
    print("Calculate begin")
    print(datetime.datetime.now())
    if dist.is_available() and dist.is_initialized():
        # Multi-GPUs
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        # Single-GPUs
        rank = 0
        world_size = 1
    
    cache_weight_file = config.get("cache_weight_file", None)
    normalization = config.get("normalization", None)
    embedding_model_name = config.get("embedding_model_name", "all-mpnet-base-v2")
    embedding_model = SentenceTransformer(embedding_model_name)
    holdout_questions = [message[0]['content'] for message in holdout_dataset['message']]
    embedding_index = embedding_question(holdout_questions, embedding_model)

    # Split training dataset
    total = len(train_dataset)
    per = total // world_size
    start = rank * per
    end = total if rank == world_size - 1 else (rank + 1) * per

    
    local_scores = []
    for train_index in range(start, end):
        target_message = train_dataset[train_index]['message']
        score = calculate_score(
            holdout_dataset['message'],
            target_message,
            embedding_index,
            embedding_model,
            model,
            tokenizer
        )
        local_scores.append(score)
        if train_index % 1000 == 0:
            print(datetime.datetime.now())
        print(start, end, train_index, score)
    
    print("Calculate done")
    print(datetime.datetime.now())

    # Merge and convert
    device = next(model.parameters()).device
    local_tensor = torch.zeros(end - start, dtype=torch.float32, device=device)
    local_tensor[:] = torch.tensor(local_scores, dtype=torch.float32, device=device)
    global_tensor = torch.zeros(total, dtype=torch.float32, device=device)
    global_tensor[start:end] = local_tensor

    if world_size > 1:
        dist.all_reduce(global_tensor, op=ReduceOp.SUM)

    # Normalizaion
    np_scores = global_tensor.cpu().numpy()
    normalization = config.get("normalization", None)
    if normalization == "min_max":
        mn = np_scores.min()
        mx = np_scores.max()
        normed_scores = (np_scores - mn) / ((mx - mn) + 1e-12)
    else:
        normed_scores = np_scores

    # Save to cache
    cache_path = config.get("cache_weight_file", None)
    if rank == 0 and cache_path:
        np.save(cache_path + ".raw.npy", np_scores)
        if normalization is not None:
            np.save(cache_path + f".{normalization}.npy", normed_scores)

    print("Done")
    print(datetime.datetime.now())
    return normed_scores