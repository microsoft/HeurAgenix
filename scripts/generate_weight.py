import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)
from importlib import import_module
import numpy as np
from trl import ModelConfig, DPOConfig, TrlParser

from alignment.configs import parse_args
from alignment.dataset_utils import load_dataset
from alignment.model_utils import get_model, get_tokenizer



def main(model_args, data_args, training_args, test_args, train_function):
    weight_args = data_args.weight_args
    cache_file = weight_args.get("cache_weight_file", os.path.join("output", "weight_cache", "weight_cache.npy"))
    if os.getenv("AMLT_DATA_DIR"):
        base_dir =  os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..")
        cache_file = os.path.join(base_dir, cache_file)

    if not os.path.exists(cache_file):
        weight_function_path = data_args.weight_function
        module, function = weight_function_path.rsplit(".", 1)
        weight_function = getattr(import_module(module), function)

        tokenizer = get_tokenizer(model_args, training_args)
        model = get_model(tokenizer, model_args, training_args)
        model.to("cuda")
        holdout_dataset, train_dataset, test_dataset = load_dataset(tokenizer, data_args)

        weight_args["train_dataset"] = train_dataset
        weight_args["holdout_dataset"] = holdout_dataset
        weight_args["model"] = model
        weight_args["tokenizer"] = tokenizer
        weights = weight_function(**weight_args)
        np.save(cache_file, weights)


if __name__ == "__main__":
    model_args, data_args, training_args, test_args, train_function = parse_args()
    main(model_args, data_args, training_args, test_args, train_function)