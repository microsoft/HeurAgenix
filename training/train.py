import sys
from datetime import datetime # Kept for potential use, though not directly in the final save path now

# Ensure stdout is flushed, can be useful in some environments
sys.stdout.flush()

# --- Local Module Imports ---
from model import get_model
from dataset import get_custom_dataset
from rewards import (
    # think_answer_consistency_reward_func,
    correctness_reward_func,
    algorithm_reward_func,
    # strict_format_reward_func,
    soft_format_reward_func,
    language_consistency_reward_func,
    cards_reward_func
)
from config import TRAINING_ARGS

# --- Library Imports ---
from trl import GRPOTrainer
from unsloth import FastLanguageModel # Though not directly instantiated, often part of Unsloth's ecosystem
import wandb # For experiment tracking
import torch.distributed as dist
import os # For path joining

# --- Main Script ---

# Load model and tokenizer
print("Loading model and tokenizer...")
model, tokenizer = get_model()
print("Model and tokenizer loaded.")

# Load custom dataset
print("Loading custom dataset...")
dataset = get_custom_dataset() # Assuming this returns the 'train' split or a DatasetDict with 'train'
print("Custom dataset loaded.")

if "wandb" in TRAINING_ARGS.report_to:
    print(f"Reporting to Weights & Biases. Project: {os.getenv('WANDB_PROJECT', 'Default')}, Run Name: {TRAINING_ARGS.run_name if TRAINING_ARGS.run_name else 'GRPO_Run'}")
else:
    print("Weights & Biases reporting is not configured in TRAINING_ARGS.report_to.")


# Initialize GRPO trainer
print("Initializing GRPOTrainer...")
trainer = GRPOTrainer(
    model=model,
    # processing_class=tokenizer, # GRPOTrainer typically doesn't use processing_class directly.
                                 # Tokenizer is often passed if model is a string ID, but here model object is passed.
                                 # If your GRPOTrainer version or setup requires it, uncomment.
                                 # Unsloth models usually embed tokenizer functionality or it's handled by the model object.
    tokenizer=tokenizer, # Pass tokenizer for prompt formatting and generation if needed by GRPOTrainer
    reward_funcs=[
        soft_format_reward_func,
        algorithm_reward_func,
        correctness_reward_func,
        language_consistency_reward_func,
        cards_reward_func
    ],
    args=TRAINING_ARGS, # GRPOConfig object
    train_dataset=dataset, # The dataset for training
    # eval_dataset=None, # Optionally, provide an evaluation dataset
)
print("GRPOTrainer initialized.")

# Start training
print("Starting training...")
trainer.train()
print("Training finished.")

# Save LoRA model adapters
final_lora_path = os.path.join(TRAINING_ARGS.output_dir, "final_lora_adapters")
print(f"Saving final LoRA adapters to: {final_lora_path}")
model.save_lora(final_lora_path)
print("LoRA adapters saved.")

# Clean up distributed process group if initialized
# This should only be called if torch.distributed.init_process_group was called.
# Unsloth or TRL/Accelerate might handle this internally.
if dist.is_initialized():
    print("Cleaning up distributed process group...")
    dist.destroy_process_group()
    print("Distributed process group destroyed.")
else:
    print("Distributed training not initialized or already cleaned up.")

print("Script finished successfully.")