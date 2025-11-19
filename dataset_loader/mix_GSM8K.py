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
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    num_proc = getattr(data_config, "dataset_process_num", None) if data_config else None
