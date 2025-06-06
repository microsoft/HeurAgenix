from datetime import datetime
import torch # Commonly used with TRL and Unsloth, even if not directly in this config block.
from unsloth import is_bfloat16_supported
from trl import GRPOConfig

# --- Model & Training Configuration ---

# Maximum sequence length for the model.
# The calculation (1024*3.5) results in 3584.
MAX_SEQ_LENGTH = int(1024*3.5)

# Rank for LoRA (Low-Rank Adaptation).
LORA_RANK = 32

TRAINING_ARGS = GRPOConfig(
    # General training parameters
    # output_dir: Directory to save model checkpoints and outputs.
    # It's recommended to customize this path for your project.
    # The example uses a timestamp for uniqueness if you prefer dynamic naming.
    output_dir=f"grpo_output_{datetime.now():%Y%m%d_%H%M%S}",
    # Or, for a static path:
    # output_dir="./grpo_checkpoints",

    learning_rate=1e-6,
    num_train_epochs=1,
    max_steps=-1, # If > 0, overrides num_train_epochs.
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",

    # Optimizer settings
    optim="paged_adamw_8bit", # Paged AdamW optimizer for 8-bit training.
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    max_grad_norm=0.1, # Gradient clipping.

    # Mixed precision training
    # bf16 enabled if supported, otherwise fp16.
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),

    # GRPO specific parameters (Generative Replay Policy Optimization)
    use_vllm=True, # Whether to use vLLM for generation. Set to False if not available or not desired.
    num_generations=12, # Number of generations for GRPO.
    max_prompt_length=2048, # Maximum length of the prompt.
    max_completion_length=256*3, # Maximum length of the generated completion.

    # Logging and saving
    logging_steps=5,
    save_steps=250, # Save a checkpoint every N steps.
    log_completions=True, # Whether to log completions.
    # report_to: List of integrations to report results to.
    # Example: ["wandb"] for Weights & Biases.
    # Remove or modify if you don't use these services or want to disable reporting.
    report_to=["wandb"],
)