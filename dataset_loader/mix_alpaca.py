from typing import Dict, List, Any
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from alignment.configs import DataConfig

def process_dataset(batch: Dict[str, List[Any]], indices: List[int], tokenizer=None) -> Dict[str, List[Any]]:
    texts = []
    instructions = batch.get("instruction", [])
    inputs       = batch.get("input", [])
    outputs      = batch.get("output", [])
    n = len(instructions)
    for i in range(n):
        instruction = (instructions[i] or "").strip()
        input = (inputs[i] or "").strip()
        if input:
            user = f"{instruction}\n\n{input}"
        else:
            user = instruction
        output = (outputs[i] or "").strip()
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": output},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    example_ids = list(indices)
    return {"text": texts, "example_id": example_ids}

def subset_map(dataset: Dataset, split_name: str, num_proc: int, tokenizer) -> Dataset:
    return dataset.map(
        process_dataset,
        batched=True,
        with_indices=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc=f"process_dataset ({split_name}) messages->text, add example_id",
    )

def get_dataset(data_config: DataConfig, tokenizer, **kwargs) -> DatasetDict:
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca_cleaned = load_dataset("yahma/alpaca-cleaned", split="train")
    num_proc = getattr(data_config, "dataset_process_num", None)

    n_rows = alpaca_cleaned.num_rows
    holdout_dataset = alpaca_cleaned.select(range(10000))
    middle_dataset = alpaca_cleaned.select(range(10000, n_rows - 10000))
    test_dataset = alpaca_cleaned.select(range(n_rows - 10000, n_rows))

    test_instructions = set(ins.strip() for ins in test_dataset["instruction"])
    filter_batch = lambda batch: [ins.strip() not in test_instructions for ins in batch["instruction"]]
    filtered_alpaca = alpaca.filter(filter_batch, num_proc=num_proc)
    train_dataset = concatenate_datasets([holdout_dataset, middle_dataset, filtered_alpaca])

    holdout_dataset = subset_map(holdout_dataset, "holdout", num_proc, tokenizer)
    train_dataset   = subset_map(train_dataset, "train", num_proc, tokenizer)
    test_dataset    = subset_map(test_dataset, "test", num_proc, tokenizer)
    return DatasetDict(
        holdout=holdout_dataset,
        train=train_dataset,
        test=test_dataset
    )