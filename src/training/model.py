from unsloth import FastLanguageModel, PatchFastRL
from config import MAX_SEQ_LENGTH, LORA_RANK # Assumes config.py contains these definitions

# Apply Patch (ensure this is done before model loading)
PatchFastRL("GRPO", FastLanguageModel)

def get_model():
    # Load pretrained model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        # model_name: Specify the Hugging Face model ID or path to your local model.
        # The original path suggested "Qwen/Qwen2.5-7B-Instruct-1M".
        # Please verify and use the correct model identifier or local path.
        model_name="path/to/your/local/model_directory", # Alternative for a truly local model

        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,   # Set to False when using 16bit LoRA, True for QLoRA with 4-bit
        fast_inference=True,  # Enable Unsloth's fast inference (uses their optimized kernels)
        # max_lora_rank: If you're loading a PEFT model with LoRA adapters directly,
        # this can be set. For adding new LoRA adapters like below, LORA_RANK is used in get_peft_model.
        # max_lora_rank=LORA_RANK, # This might be redundant if get_peft_model is used next
        gpu_memory_utilization=0.6, # Adjust if out of memory, e.g., 0.9 for more utilization
        # device_map="auto", # Usually handled by Unsloth, but can be specified if needed
    )

    # Apply LoRA configuration to the model
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Target modules for LoRA. Adjust if memory is an issue or for different model architectures.
        lora_alpha=LORA_RANK, # Typically same as r or 2*r
        use_gradient_checkpointing="unsloth", # Recommended by Unsloth for long context fine-tuning
        random_state=3407, # For reproducibility
    )
    return model, tokenizer