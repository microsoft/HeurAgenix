import os
import torch
import numpy as np
from trl import DataCollatorForCompletionOnlyLM
from importlib import import_module

class KeepKeysWrapper:
    def __init__(self, base_collator, keep_key="example_id"):
        self.base_collator = base_collator
        self.keep_key = keep_key

    def __call__(self, features):
        ids = None
        if self.keep_key in features[0]:
            ids = torch.tensor([f.pop(self.keep_key) for f in features], dtype=torch.long)
        for f in features:
            f.pop("message", None)
            f.pop("text", None)

        batch = self.base_collator(features)
        if ids is not None:
            batch[self.keep_key] = ids
        return batch


class EoTCompletionCollator:
    def __init__(self, base_collator, tokenizer):
        self.base = base_collator
        self.eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    def __call__(self, features):
        batch = self.base(features)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        B, S = labels.shape
        for i in range(B):
            sup = (labels[i] != -100).nonzero(as_tuple=True)[0]
            if sup.numel() == 0:
                continue
            last = sup[-1].item()
            if last + 1 < S and input_ids[i, last + 1].item() == self.eot_id:
                labels[i, last + 1] = self.eot_id
        batch["labels"] = labels
        return batch


def load_dataset(tokenizer, data_args):
    dataset_loader_path = data_args.dataset_loader
    module, function = dataset_loader_path.rsplit(".", 1)
    dataset_loader = getattr(import_module(module), function)
    dataset = dataset_loader(data_args, tokenizer)
    holdout_dataset, train_dataset, test_dataset = \
        dataset[data_args.dataset_holdout_split], dataset[data_args.dataset_train_split], dataset[data_args.dataset_test_split]
    return holdout_dataset, train_dataset, test_dataset


def load_weight(train_dataset, holdout_dataset, model, tokenizer, data_args):
    weight_function_path = data_args.weight_function
    weight_args = data_args.weight_args
    module, function = weight_function_path.rsplit(".", 1)
    weight_function = getattr(import_module(module), function)

    if weight_args and os.path.exists(weight_args.get("cache_weight_file", None)):
        weights = np.load(weight_args.get("cache_weight_file", None))
    else:
        weight_args["train_dataset"] = train_dataset
        weight_args["holdout_dataset"] = holdout_dataset
        weight_args["model"] = model
        weight_args["tokenizer"] = tokenizer
        weights = weight_function(**weight_args)
    return weights


def infer_response_template(tokenizer):
    messages = [{"role": "user", "content": "test"}]
    enc0 = tokenizer(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False
        ),
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    enc1 = tokenizer(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        ),
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    suffix = enc1.input_ids[0, enc0.input_ids.shape[1]:]
    response_template_id = suffix.tolist()
    response_template = tokenizer.decode(suffix.tolist(), skip_special_tokens=False)
    return response_template, response_template_id


def get_data_collator(tokenizer):
    response_template, response_template_id = infer_response_template(tokenizer)
    try:
        base= DataCollatorForCompletionOnlyLM(response_template_id=response_template_id, tokenizer=tokenizer)
    except TypeError:
        base= DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
    base = EoTCompletionCollator(base, tokenizer)
    data_collator = KeepKeysWrapper(base_collator=base, keep_key="example_id")
    return data_collator