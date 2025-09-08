import os
from typing import Dict, List, Any
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from alignment.configs import DataConfig

def process_dataset(batch, indices, tokenizer=None, ):
    prompts = []
    chosens = []
    rejecteds = []
    example_ids = list(indices)

    prompts_raw = batch.get("prompt", None)
    chosen_messages = batch["chosen"]
    rejected_messages = batch["rejected"]

    n = len(chosen_messages)
    for i in range(n):
        if prompts_raw is not None and prompts_raw[i] is not None:
            user_text = (prompts_raw[i] or "").strip()
        else:
            cm = chosen_messages[i]
            user_turns = [m["content"] for m in cm if m.get("role") == "user"]
            user_text = user_turns[-1].strip() if len(user_turns) > 0 else ""

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

        cm = chosen_messages[i]
        rm = rejected_messages[i]
        ch_ass = [m["content"] for m in cm if m.get("role") == "assistant"]
        rj_ass = [m["content"] for m in rm if m.get("role") == "assistant"]
        ch_text = ch_ass[-1].strip() if len(ch_ass) > 0 else ""
        rj_text = rj_ass[-1].strip() if len(rj_ass) > 0 else ""
        chosens.append(ch_text)
        rejecteds.append(rj_text)

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


def get_dataset(data_config: DataConfig, tokenizer, **kwargs) -> DatasetDict:
    num_proc = getattr(data_config, "dataset_process_num", None)

    if os.getenv("AMLT_DATA_DIR"):
        dataset_base_dir = os.path.join(os.getenv("AMLT_DATA_DIR"), "dataset")
        raw_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", cache_dir=dataset_base_dir)
    else:
        raw_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")

    def is_holdout(chosen_score, reject_score):
        return (chosen_score is not None) and (reject_score is not None) and (chosen_score >= 8.5 and reject_score >= 6.5)
    holdout_dataset = train_dataset.filter(is_holdout, input_columns=["score_chosen", "score_rejected"])
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