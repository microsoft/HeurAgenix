from typing import Dict, List, Any
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from alignment.configs import DataConfig


def process_dataset(batch: Dict[str, List[Any]], indices: List[int], tokenizer=None) -> Dict[str, List[Any]]:
    texts       = []
    messages    = []
    example_ids = list(indices)

    for i in range(len(example_ids)):
        question    = batch["question"][i]
        answer      = batch["answer"][i]
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


def get_dataset(tokenizer, data_config: DataConfig=None, **kwargs) -> DatasetDict:
    gsm8k_train = load_dataset("openai/gsm8k", "main", split="train")
    gsm8k_noise = load_dataset("VictorYXL/noise_gsm8k", split="noise")
    gsm8k_test  = load_dataset("openai/gsm8k", "main", split="test")

    holdout_dataset = gsm8k_train.select(range(0, 1000))
    train_dataset   = gsm8k_noise.select(range(1000, gsm8k_noise.num_rows))
    train_dataset   = concatenate_datasets([holdout_dataset, train_dataset])
    test_dataset    = gsm8k_test
    num_proc = getattr(data_config, "dataset_process_num", None) if data_config else None

    holdout_dataset = subset_map(holdout_dataset, "holdout", num_proc, tokenizer)
    train_dataset   = subset_map(train_dataset, "train", num_proc, tokenizer)
    test_dataset    = subset_map(test_dataset, "test", num_proc, tokenizer)
    return DatasetDict(
        holdout=holdout_dataset,
        train=train_dataset,
        test=test_dataset
    )