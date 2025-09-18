import os
import numpy as np
from typing import Dict, List, Any
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from alignment.configs import DataConfig

def process_dataset(batch: Dict[str, List[Any]], indices: List[int], tokenizer=None) -> Dict[str, List[Any]]:
    prompt_texts      = []
    prompt_messages   = []
    chosen_messages   = []
    rejected_messages = []
    chosen_answers    = []
    rejected_answers  = []
    example_ids       = list(indices)

    for i in range(len(example_ids)):
        question  = batch["history"][i]
        a_text    = batch["human_ref_A"][i]
        b_text    = batch["human_ref_B"][i]
        labels    = batch["labels"][i]

        chosen_answer, rejected_answer = (a_text, b_text) if labels == 1 else (b_text, a_text)

        prompt_message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": question},
        ]
        chosen_message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": question},
            {"role": "assistant", "content": chosen_answer},
        ]
        rejected_message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": question},
            {"role": "assistant", "content": rejected_answer},
        ]

        prompt_messages.append(prompt_message)
        chosen_messages.append(chosen_message)
        rejected_messages.append(rejected_message)

        prompt_text = tokenizer.apply_chat_template(
            prompt_message,
            add_generation_prompt=True,
            tokenize=False
        )
        prompt_texts.append(prompt_text)
        chosen_answers.append(chosen_answer)
        rejected_answers.append(rejected_answer)

    return {
        "chosen_message": chosen_messages,
        "rejected_message": rejected_messages,
        "prompt": prompt_texts,
        "chosen": chosen_answers,
        "rejected": rejected_answers,
        "example_id": example_ids,
    }

def subset_map(dataset: Dataset, split_name: str, num_proc: int, tokenizer) -> Dataset:
    return dataset.map(
        process_dataset,
        batched=True,
        with_indices=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc=f"process_dataset ({split_name}) messages->text, add example_id",
        load_from_cache_file=False,
    )

def take_first_n_per_class(dataset: Dataset, number_per_topic: int = 1000) -> Dataset:
    domains = dataset["domain"]

    unique, first_idx, counts = np.unique(domains, return_index=True, return_counts=True)
    mask = counts >= 1000
    starts = first_idx[mask]
    starts = np.sort(starts)
    indices = np.concatenate([np.arange(s, s + 1000) for s in starts]).tolist()
    sub_dataset = dataset.select(indices)

    return sub_dataset

def get_dataset(tokenizer, data_config: DataConfig=None, **kwargs) -> DatasetDict:
    num_proc = getattr(data_config, "dataset_process_num", None) if data_config else None

    raw_dataset = load_dataset("stanfordnlp/SHP-2")

    train_subset = take_first_n_per_class(raw_dataset["train"], 1000)
    target_topic = "askscience_test"
    target_test_set = raw_dataset["test"].filter(lambda ex: ex["domain"] == target_topic)
    target_test_set = target_test_set.map( 
        lambda batch: {"sum_score": [a + b for a, b in zip(batch["score_A"], batch["score_B"])]},  
        batched=True  
    ).sort("sum_score", reverse=True)

    holdout_dataset = target_test_set.select(range(0, 1000))
    train_dataset   = concatenate_datasets([holdout_dataset, train_subset])
    test_dataset    = target_test_set.select(range(1000, 2000))

    holdout_dataset = subset_map(holdout_dataset, "holdout", num_proc, tokenizer)
    train_dataset   = subset_map(train_dataset, "train", num_proc, tokenizer)
    test_dataset    = subset_map(test_dataset, "test", num_proc, tokenizer)

    return DatasetDict(
        holdout=holdout_dataset,
        train=train_dataset,
        test=test_dataset
    )