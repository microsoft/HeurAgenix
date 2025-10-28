from unsloth import FastLanguageModel, PatchFastRL
from config import MAX_SEQ_LENGTH, LORA_RANK # Assumes config.py contains these definitions

PatchFastRL("GRPO", FastLanguageModel)

def get_model():
    # Load pretrained model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="path/to/your/local/model_directory", # Alternative for a truly local model

        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,   # Set to False when using 16bit LoRA, True for QLoRA with 4-bit
        fast_inference=True,  # Enable Unsloth's fast inference (uses their optimized kernels)
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
        ],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth", # Recommended by Unsloth for long context fine-tuning
    )
    return model, tokenizer