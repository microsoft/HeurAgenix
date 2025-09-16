import os
from typing import Dict, List, Any
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from alignment.configs import DataConfig

def process_dataset(batch: Dict[str, List[Any]], indices: List[int], tokenizer=None) -> Dict[str, List[Any]]:
    texts       = []
    messages    = []
    example_ids = list(indices)
    
    for i in range(len(example_ids)):
        instruction = batch["question_title"][i]
        question    = batch["question_content"][i]
        answer      = batch["best_answer"][i]
        if question:
            question = f"{instruction}\n\n{question}"
        else:
            question = instruction
        message = [
            {"role":"system","content":"You are a helpful assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        messages.append(message)
        texts.append(tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False))
    return {"message": messages, "text": texts, "example_id": example_ids}


def take_first_n_per_class(dataset: Dataset, number_per_topic: int = 10000) -> Dataset:

    num_topics = len(dataset.features["topic"].names)
    counters = [0] * num_topics
    completed = 0
    indices = []

    for idx, ex in enumerate(dataset):
        label_id = ex["topic"]
        if counters[label_id] < number_per_topic:
            indices.append(idx)
            counters[label_id] += 1
            if counters[label_id] == number_per_topic:
                completed += 1
                if completed == num_topics:
                    break
    sub_dataset = dataset.select(indices)
    return sub_dataset

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


def get_dataset(data_config: DataConfig, tokenizer, **kwargs) -> DatasetDict:
    num_proc = getattr(data_config, "dataset_process_num", None) if data_config else None

    if os.getenv("AMLT_DATA_DIR"):
        dataset_base_dir = os.path.join(os.getenv("AMLT_DATA_DIR"), "dataset")
        raw_dataset = load_dataset("community-datasets/yahoo_answers_topics", cache_dir=dataset_base_dir)
    else:
        raw_dataset = load_dataset("community-datasets/yahoo_answers_topics")

    target_topic = "Sports"
    topic_id = raw_dataset["test"].features["topic"].str2int(target_topic)
    target_test_set = raw_dataset["test"].filter(lambda ex: ex["topic"] == topic_id)
    train_subset = take_first_n_per_class(raw_dataset["train"], 10000)

    holdout_dataset = target_test_set.select(range(0, 3000))
    train_dataset   = concatenate_datasets([holdout_dataset, train_subset])
    test_dataset    = target_test_set.select(range(3000, 6000))

    holdout_dataset = subset_map(holdout_dataset, "holdout", num_proc, tokenizer)
    train_dataset   = subset_map(train_dataset, "train", num_proc, tokenizer)
    test_dataset    = subset_map(test_dataset, "test", num_proc, tokenizer)

    return DatasetDict(
        holdout=holdout_dataset,
        train=train_dataset,
        test=test_dataset
    )