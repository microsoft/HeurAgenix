import os
from typing import Dict, List, Any
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from alignment.configs import DataConfig

def process_dataset(batch, indices, tokenizer=None):
    chosen_messages = []
    rejected_messages = []
    chosen_answers = []
    rejected_answers = []
    example_ids = list(indices)

    for i in range(len(example_ids)):
        question        = batch["prompt"][i]
        chosen_answer   = batch["chosen"][i][-1]["content"]
        rejected_answer = batch["rejected"][i][-1]["content"]

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

        chosen_messages.append(chosen_message)
        rejected_messages.append(rejected_message)

        chosen_answer = tokenizer.apply_chat_template(
            chosen_message,
            add_generation_prompt=False,
            tokenize=False
        )
        rejected_answer = tokenizer.apply_chat_template(
            rejected_message,
            add_generation_prompt=False,
            tokenize=False
        )
        chosen_answers.append(chosen_answer)
        rejected_answers.append(rejected_answer)

    return {
        "chosen_message": chosen_messages,
        "rejected_message": rejected_messages,
        "chosen_answer": chosen_answers,
        "rejected_answer": rejected_answers,
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


def get_dataset(data_config: DataConfig, tokenizer, **kwargs) -> DatasetDict:
    num_proc = getattr(data_config, "dataset_process_num", None)

    if os.getenv("AMLT_DATA_DIR"):
        dataset_base_dir = os.path.join(os.getenv("AMLT_DATA_DIR"), "dataset")
        raw_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", cache_dir=dataset_base_dir)
    else:
        raw_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")

    def is_holdout(chosen_score, reject_score):
        return (chosen_score is not None) and (reject_score is not None) and (chosen_score >= 9 and reject_score >= 7)
    holdout_dataset = raw_dataset["train_prefs"].filter(is_holdout, input_columns=["score_chosen", "score_rejected"])
    train_dataset   = concatenate_datasets([holdout_dataset, raw_dataset["train_prefs"]])
    test_dataset    = raw_dataset["test_prefs"]

    holdout_dataset = subset_map(holdout_dataset, "holdout", num_proc, tokenizer)
    train_dataset   = subset_map(train_dataset, "train", num_proc, tokenizer)
    test_dataset    = subset_map(test_dataset, "test", num_proc, tokenizer)

    return DatasetDict(
        holdout=holdout_dataset,
        train=train_dataset,
        test=test_dataset
    )