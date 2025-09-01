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