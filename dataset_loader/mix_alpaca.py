from datasets import load_dataset, DatasetDict, concatenate_datasets
from alignment.configs import DataConfig

def to_messages(example):
    instruction = (example.get("instruction") or "").strip()
    input = (example.get("input") or "").strip()
    if input:
        user = f"{instruction}\n\n{input}"
    else:
        user = instruction
    output = (example.get("output") or "").strip()
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": output},
        ]
    }

def only_messages(dataset):
    columns_to_remove = [column for column in dataset.column_names if column != "messages"]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)
    return dataset

def get_dataset(data_config: DataConfig) -> DatasetDict:
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca_cleaned = load_dataset("yahma/alpaca-cleaned", split="train")
    num_proc = getattr(data_config, "dataset_num_proc", None)


    alpaca = only_messages(alpaca.map(
        to_messages,
        remove_columns=alpaca.column_names,
        num_proc=num_proc,
    ))
    alpaca_cleaned = only_messages(alpaca_cleaned.map(
        to_messages,
        remove_columns=alpaca_cleaned.column_names,
        num_proc=num_proc,
    ))

    holdout_num = min(10000, alpaca_cleaned.num_rows)
    holdout_dataset = alpaca_cleaned.select(range(holdout_num))
    train_dataset = concatenate_datasets([alpaca, holdout_dataset]).shuffle(seed=42)
    test_dataset = alpaca_cleaned.select(range(holdout_num, alpaca_cleaned.num_rows)) if alpaca_cleaned.num_rows > holdout_num else alpaca_cleaned.select([])
    return DatasetDict(
        holdout=holdout_dataset,
        train=train_dataset,
        test=test_dataset)