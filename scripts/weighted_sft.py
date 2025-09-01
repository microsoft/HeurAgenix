import logging
import os
import sys

import datasets
import transformers
import numpy as np
from importlib import import_module
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, TrlParser, get_peft_config, setup_chat_format

from alignment import SFTConfig, DataConfig, get_model, get_tokenizer
from scripts.weighted_trainers import KeepKeysCollator, WeightedSFTTrainer


logger = logging.getLogger(__name__)

def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")


    ############
    # Load model
    ############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None or model.config.pad_token_id == -1:
        model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    if tokenizer.chat_template is None:
        logger.info("No chat template provided, using ChatML.")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    ################
    # Load datasets
    ################
    dataset_loader_path = data_args.dataset_loader
    logger.info(f"Loading dataset via custom loader: {dataset_loader_path}")
    module, function = dataset_loader_path.rsplit(".", 1)
    dataset_loader = getattr(import_module(module), function)
    dataset = dataset_loader(data_args, tokenizer)
    holdout_dataset, train_dataset, test_dataset = \
        dataset[data_args.dataset_holdout_split], dataset[data_args.dataset_train_split], dataset[data_args.dataset_test_split]
    dataset_num_proc = getattr(data_args, "dataset_process_num", None)
    weight_function_path = data_args.weight_function
    weight_args = data_args.weight_args
    logger.info(f"Calculate weight by {weight_function_path} with args: {weight_args}")
    module, function = weight_function_path.rsplit(".", 1)
    weight_function = getattr(import_module(module), function)
    if os.path.exists(weight_args.get("cache_weight_file", None)):
        weights = np.load(weight_args.get("cache_weight_file", None))
    else:
        weight_args["train_dataset"] = train_dataset
        weight_args["holdout_dataset"] = holdout_dataset
        weight_args["model"] = model
        weight_args["tokenizer"] = tokenizer
        weights = weight_function(**weight_args)



    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        holdout_dataset=holdout_dataset,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
        dataset_text_field="text",
        packing=False,
        max_seq_length=training_args.max_seq_length,
        dataset_num_proc=dataset_num_proc,
        weight_function=weight_function,
        weight_args=weight_args,
        weights=weights
    )


    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.model.config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "model_name": training_args.hub_model_id if training_args.push_to_hub else None,
        "dataset_loader": data_args.dataset_loader,
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.eval_function:
        logger.info("*** Evaluate ***")
        eval_path = training_args.eval_function
        eval_args = training_args.eval_args
        logger.info(f"Evaluate by {eval_path}")
        module, function = eval_path.rsplit(".", 1)
        eval_function = getattr(import_module(module), function)
        eval_args["model"] = trainer.model
        eval_args["tokenizer"] = tokenizer
        eval_args["test_dataset"] = test_dataset
        eval_args["output_dir"] = training_args.output_dir
        metrics = eval_function(**eval_args)


if __name__ == "__main__":
    parser = TrlParser((ModelConfig, DataConfig, SFTConfig))
    model_args, data_args, training_args = parser.parse_args_and_config()
    main(model_args, data_args, training_args)
