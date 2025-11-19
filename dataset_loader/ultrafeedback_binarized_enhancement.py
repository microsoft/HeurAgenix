from typing import Dict, List, Any
from datasets import load_dataset, DatasetDict, Dataset
from alignment.configs import DataConfig

def process_dataset(batch: Dict[str, List[Any]], indices: List[int], tokenizer=None) -> Dict[str, List[Any]]:
    prompt_texts      = []
    prompt_messages   = []
    chosen_messages   = []
    rejected_messages = []
    chosen_answers    = []
    rejected_answers  = []
    example_ids       = list(indices)

    for i in range(len(example_ids)):
        question        = batch["prompt"][i]
        chosen_answer   = batch["chosen"][i][-1]["content"]
        rejected_answer = batch["rejected"][i][-1]["content"]

        prompt_message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": question},
        ]
        chosen_message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": question},
            {"role": "assistant", "content": chosen_answer},
        ]
        rejected_meesage = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": question},
            {"role": "assistant", "content": rejected_answer},
        ]

        prompt_messages.append(prompt_message)
        chosen_messages.append(chosen_message)
        rejected_messages.append(rejected_meesage)

        prompt_text = tokenizer.apply_chat_template(
            prompt_message,
            add_generation_prompt=True,
            tokenize=False
        )
        prompt_texts.append(prompt_text)
        chosen_answers.append(chosen_answer)
        rejected_answers.append(rejected_answer)

    return {
        "chosen_message": chosen_messages,
        "rejected_message": rejected_messages,
        "prompt": prompt_texts,
        "chosen": chosen_answers,
        "rejected": rejected_answers,
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


def get_dataset(tokenizer, data_config: DataConfig=None, **kwargs) -> DatasetDict:
    num_proc = getattr(data_config, "dataset_process_num", None) if data_config else None
    raw_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")

    def is_holdout(chosen_score, reject_score):
        return (chosen_score is not None) and (reject_score is not None) and (chosen_score >= 9 and reject_score >= 7)
    train_dataset   = raw_dataset["train_prefs"]
    holdout_dataset = train_dataset.filter(is_holdout, input_columns=["score_chosen", "score_rejected"])
    test_dataset    = raw_dataset["test_prefs"]

    holdout_dataset = subset_map(holdout_dataset, "holdout", num_proc, tokenizer)
    train_dataset   = subset_map(train_dataset, "train", num_proc, tokenizer)
    test_dataset    = subset_map(test_dataset, "test", num_proc, tokenizer)

    return DatasetDict(
        holdout=holdout_dataset,
        train=train_dataset,
        test=test_dataset
    )