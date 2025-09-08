import os
import numpy as np
from importlib import import_module


def load_dataset(tokenizer, data_args):
    dataset_loader_path = data_args.dataset_loader
    module, function = dataset_loader_path.rsplit(".", 1)
    dataset_loader = getattr(import_module(module), function)
    dataset = dataset_loader(data_args, tokenizer)
    holdout_dataset, train_dataset, test_dataset = \
        dataset[data_args.dataset_holdout_split], dataset[data_args.dataset_train_split], dataset[data_args.dataset_test_split]
    return holdout_dataset, train_dataset, test_dataset


def load_weight(train_dataset, holdout_dataset, model, tokenizer, data_args):
    # TODO: current load from cache only, need to support no cache
    weight_function_path = data_args.weight_function
    weight_args = data_args.weight_args

    cache_file = weight_args.get("cache_weight_file", os.path.join("output", "weight_cache", "weight_cache.npy"))
    if os.getenv("AMLT_DATA_DIR"):
        base_dir =  os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..")
        cache_file = os.path.join(base_dir, cache_file)
    normalization = weight_args.get("normalization", None)

    assert os.path.exists(cache_file), "Cache file does not exist"
    weights = np.load(cache_file)
        
    if normalization == "min_max":
        mn = float(weights.min())
        mx = float(weights.max())
        normed_weights = (weights - mn) / (mx - mn + 1e-12)
    else:
        normed_weights = weights

    return normed_weights


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
