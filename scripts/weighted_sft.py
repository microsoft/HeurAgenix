import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)
import datasets
import transformers
import torch.distributed as dist
from alignment.configs import DataConfig, TestConfig
from alignment.dataset_utils import get_data_collator, load_dataset, load_weight
from alignment.log import get_log
from alignment.model_utils import get_model, get_tokenizer
from scripts.weighted_sft_trainers import WeightedSFTTrainer
from transformers import set_seed
from trl import ModelConfig, SFTConfig, TrlParser, get_peft_config
import wandb


def main(model_args, data_args, training_args, test_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    if os.getenv("AMLT_DATA_DIR"):
        base_dir =  os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..")
        training_args.output_dir = os.path.join(base_dir, training_args.output_dir)
        model_dir = os.path.join(os.getenv("AMLT_DATA_DIR"), "model_cache")
        model_args.model_name_or_path = os.path.join(model_dir, model_args.model_name_or_path.replace("/", "___"))
    os.makedirs(training_args.output_dir, exist_ok=True)
    logger = get_log(os.path.join(training_args.output_dir, "log.txt"))
    if not int(os.environ.get("RANK", "0")) == 0:
        logger.disabled = True

    datasets.utils.logging.set_verbosity(logger.level)
    transformers.utils.logging.set_verbosity(logger.level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training parameters {training_args}")

    ############
    # Load tokenizer and model
    ############
    logger.info("*** Loading model ***")
    tokenizer = get_tokenizer(model_args, training_args)
    model = get_model(tokenizer, model_args, training_args)


    ################
    # Load datasets
    ################
    logger.info(f"Loading dataset via custom loader: {data_args.dataset_loader}")
    holdout_dataset, train_dataset, test_dataset = load_dataset(tokenizer, data_args)
    logger.info(f"Calculate weight by {data_args.weight_function} with args: {data_args.weight_args}")
    weights = load_weight(train_dataset, holdout_dataset, model, tokenizer, data_args)

    ############################
    # Initialize the SFT Trainer
    ############################
    data_collator = get_data_collator(tokenizer)
    trainer = WeightedSFTTrainer(
        weights=weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
        dataset_text_field="text",
        packing=False,
        max_seq_length=training_args.max_seq_length,
        dataset_num_proc = getattr(data_args, "dataset_process_num", None),
        data_collator=data_collator,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    train_result = trainer.train()
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
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    trainer.accelerator.wait_for_everyone()

    if trainer.accelerator.is_main_process:
        # Save everything else on main process
        kwargs = {
            "model_name": training_args.hub_model_id if training_args.push_to_hub else None,
            "tags": ["alignment-handbook"],
        }
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

        try:
            if trainer.accelerator.is_main_process:
                wandb.finish(quiet=True)
            trainer.accelerator.end_training()
        except Exception:
            pass
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

if __name__ == "__main__":
    parser = TrlParser((ModelConfig, DataConfig, SFTConfig, TestConfig))
    model_args, data_args, training_args, test_args = parser.parse_args_and_config()
    main(model_args, data_args, training_args, test_args)