import os
import os, sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)
import torch
import torch.distributed as dist
from importlib import import_module

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ModelConfig, TrlParser
from alignment.configs import SFTConfig, DataConfig, TestConfig
from alignment.dataset_utils import load_dataset
from alignment.log import get_log
import argparse


def init_dist_if_needed(force_distributed: bool | None = None):
    if force_distributed is None:
        need_dist = int(os.getenv("WORLD_SIZE", "1")) > 1
    else:
        need_dist = force_distributed

    if need_dist:
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.getenv("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0

def main(model_args, data_args, training_args, test_args):
    os.makedirs(training_args.output_dir, exist_ok=True)
    logger = get_log(os.path.join(training_args.output_dir, "log.txt"))

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--output_dir", type=str, default=None)
    extra, _ = ap.parse_known_args()
    output_dir = extra.output_dir or training_args.output_dir
    assert output_dir and os.path.isdir(output_dir), f"Output dir not found: {output_dir}"

    is_dist, rank, world_size, local_rank = init_dist_if_needed()

    ################
    # Load tokenizer and model
    ################
    tokenizer = AutoTokenizer.from_pretrained(output_dir, use_fast=True, trust_remote_code=model_args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if model_args.torch_dtype in (None, "auto") else getattr(torch, model_args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map={"": local_rank},
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=model_args.trust_remote_code,
    )
    model.eval()
    model.config.use_cache = True

    ################
    # Load datasets
    ################
    logger.info(f"Loading dataset via custom loader: {data_args.dataset_loader}")
    holdout_dataset, train_dataset, test_dataset = load_dataset(tokenizer, data_args)

    ##########
    # Evaluate
    ##########
    logger.info("*** Evaluate ***")
    eval_path = test_args.eval_function
    eval_args = test_args.eval_args
    logger.info(f"Evaluate by {eval_path}")
    module, function = eval_path.rsplit(".", 1)
    eval_function = getattr(import_module(module), function)
    eval_args["model"] = model
    eval_args["tokenizer"] = tokenizer
    eval_args["test_dataset"] = test_dataset
    eval_args["output_file"] = os.path.join(output_dir, "test_results.json") if rank == 0 else None
    metrics = eval_function(**eval_args)
    logger.info(f"Evaluate result: {metrics}")


if __name__ == "__main__":
    parser = TrlParser((ModelConfig, DataConfig, SFTConfig, TestConfig))
    model_args, data_args, training_args, test_args = parser.parse_args_and_config()
    main(model_args, data_args, training_args, test_args)
