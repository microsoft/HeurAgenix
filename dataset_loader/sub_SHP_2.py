import os
import numpy as np
from collections import Counter, defaultdict
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from alignment.configs import DataConfig

def process_dataset(batch, indices, tokenizer=None):
    prompts = []
    chosens = []
    rejecteds = []
    example_ids = list(indices)

    histories = batch.get("history", None)
    refs_A = batch["human_ref_A"]
    refs_B = batch["human_ref_B"]
    labels = batch["labels"]

    n = len(labels)
    for i in range(n):
        user_text = (histories[i] or "").strip() if histories is not None else ""
        messages_list = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": user_text},
        ]
        prompt = tokenizer.apply_chat_template(
            messages_list,
            add_generation_prompt=True,
            tokenize=False
        )
        prompts.append(prompt)

        a_text = (refs_A[i] or "").strip()
        b_text = (refs_B[i] or "").strip()
        if labels[i] == 1:
            chosens.append(a_text)
            rejecteds.append(b_text)
        else:
            chosens.append(b_text)
            rejecteds.append(a_text)

    return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds, "example_id": example_ids}

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