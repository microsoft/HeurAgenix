import os
from typing import Dict, List, Any
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from alignment.configs import DataConfig

def process_dataset(batch: Dict[str, List[Any]], indices: List[int], tokenizer=None) -> Dict[str, List[Any]]:
    texts       = []
    messages    = []
    example_ids = list(indices)

    for i in range(len(example_ids)):
        instruction = batch["instruction"][i]
        question    = batch["input"][i]
        answer      = batch["output"][i]
        instruction = (instruction or "").strip()
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

def get_dataset(tokenizer, data_config: DataConfig=None , **kwargs) -> DatasetDict:
    if os.getenv("AMLT_DATA_DIR"):
        dataset_base_dir = os.path.join(os.getenv("AMLT_DATA_DIR"), "dataset")
        alpaca = load_dataset("tatsu-lab/alpaca", split="train", cache_dir=dataset_base_dir)
        alpaca_cleaned = load_dataset("yahma/alpaca-cleaned", split="train", cache_dir=dataset_base_dir)
    else:
        alpaca = load_dataset("tatsu-lab/alpaca", split="train")
        alpaca_cleaned = load_dataset("yahma/alpaca-cleaned", split="train")
    num_proc = getattr(data_config, "dataset_process_num", None) if data_config else None

    n_rows = alpaca_cleaned.num_rows
    holdout_dataset = alpaca_cleaned.select(range(10000))
    middle_dataset = alpaca_cleaned.select(range(10000, n_rows - 10000))
    test_dataset = alpaca_cleaned.select(range(n_rows - 10000, n_rows))

    test_instructions = set(ins.strip() for ins in test_dataset["instruction"])
    filtered_indices = [index for index, instruction in enumerate(alpaca["instruction"]) if instruction.strip() not in test_instructions]
    filtered_alpaca = alpaca.select(filtered_indices)
    train_dataset = concatenate_datasets([holdout_dataset, middle_dataset, filtered_alpaca])

    holdout_dataset = subset_map(holdout_dataset, "holdout", num_proc, tokenizer)
    train_dataset   = subset_map(train_dataset, "train", num_proc, tokenizer)
    test_dataset    = subset_map(test_dataset, "test", num_proc, tokenizer)
    return DatasetDict(
        holdout=holdout_dataset,
        train=train_dataset,
        test=test_dataset
    )