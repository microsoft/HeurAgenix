from peft import PeftModel
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from configs import parse_args
from trl import ModelConfig, SFTConfig, get_kbit_device_map, get_quantization_config

def merge_lora_adapter(model_args: ModelConfig, training_args: SFTConfig):
    adapter_dir = training_args.output_dir
    lora_adapter_merged_path = os.path.join(training_args.output_dir, "lora_merged")
    os.makedirs(lora_adapter_merged_path, exist_ok=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)

    # Attach adapter to the base model
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    # Merge LoRA weights into the base model and unload adapter modules
    merged = peft_model.merge_and_unload(safe_merge=True)

    # Save a full, standalone model (no PEFT needed at inference)
    merged.save_pretrained(lora_adapter_merged_path)
    tokenizer.save_pretrained(lora_adapter_merged_path)

if __name__ == "__main__":
    model_args, data_args, training_args, test_args, train_function = parse_args()
    merge_lora_adapter(model_args, training_args)