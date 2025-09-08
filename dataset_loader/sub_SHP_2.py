import os
import numpy as np
from collections import Counter, defaultdict
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from alignment.configs import DataConfig

def process_dataset(batch, indices, tokenizer=None):
    chosen_messages = []
    rejected_messages = []
    chosen_texts = []
    rejected_texts = []
    example_ids = list(indices)

    for i in range(len(example_ids)):
        question  = batch["history"][i]
        a_text    = batch["human_ref_A"][i]
        b_text    = batch["human_ref_B"][i]
        labels    = batch["labels"][i]

        chosen_text, rejected_text = (a_text, b_text) if labels == 1 else (b_text, a_text)

        postive_message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": question},
            {"role": "assistant", "content": chosen_text},
        ]
        rejected_meesage = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": question},
            {"role": "assistant", "content": rejected_text},
        ]

        chosen_messages.append(postive_message)
        rejected_messages.append(rejected_meesage)

        chosen_text = tokenizer.apply_chat_template(
            postive_message,
            add_generation_prompt=False,
            tokenize=False
        )
        rejected_text = tokenizer.apply_chat_template(
            rejected_meesage,
            add_generation_prompt=False,
            tokenize=False
        )
        chosen_texts.append(chosen_text)
        rejected_texts.append(rejected_text)

    return {
        "chosen_message": chosen_messages,
        "rejected_message": rejected_messages,
        "chosen_text": chosen_texts,
        "rejected_text": rejected_texts,
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

def get_dataset(data_config: DataConfig, tokenizer, **kwargs) -> DatasetDict:
    num_proc = getattr(data_config, "dataset_process_num", None)

    if os.getenv("AMLT_DATA_DIR"):
        dataset_base_dir = os.path.join(os.getenv("AMLT_DATA_DIR"), "dataset")
        raw_dataset = load_dataset("stanfordnlp/SHP-2", cache_dir=dataset_base_dir)
    else:
        raw_dataset = load_dataset("stanfordnlp/SHP-2")

    train_subset = take_first_n_per_class(raw_dataset["train"], 1000)
    target_topic = "askbaking_test"
    target_test_set = raw_dataset["test"].filter(lambda ex: ex["domain"] == target_topic)

    holdout_dataset = target_test_set.select(range(0, 1000))
    train_dataset   = concatenate_datasets([holdout_dataset, train_subset])
    test_dataset    = target_test_set.select(range(1000, 2429))

    holdout_dataset = subset_map(holdout_dataset, "holdout", num_proc, tokenizer)
    train_dataset   = subset_map(train_dataset, "train", num_proc, tokenizer)
    test_dataset    = subset_map(test_dataset, "test", num_proc, tokenizer)

    return DatasetDict(
        holdout=holdout_dataset,
        train=train_dataset,
        test=test_dataset
    )