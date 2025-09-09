# Weighted Training

A lightweight framework to train instruction-following models with per-example weights. It supports:
- Supervised Fine-Tuning (SFT)
- Direct Preference Optimization (DPO)
- SimPO (work-in-progress)
- Weighting mechanisms that can be computed offline (e.g., via a holdout set) and applied during training

The project provides dataset loaders, weighted trainers, evaluation utilities (Azure GPT-based), and recipes to reproduce experiments with different models and datasets.

## Features
- Weighted SFT and DPO training
- SimPO support (under testing and adjustments)
- Per-example ID tracking and custom collators (EoTCompletionCollator) compatible with TRL 0.9.6
- Response template inference for chat models (e.g., Llama-3 Instruct)
- DeepSpeed ZeRO-3 acceleration via Accelerate
- Config-driven recipes for common tasks

## Repository Structure

| Path | Description |
|------|-------------|
| environment.yaml | Conda environment definition |
| run.sh | One-click script to run the full pipeline |
| alignment/configs.py | Config parsing utilities |
| alignment/dataset_utils.py | Dataset helper utilities |
| alignment/log.py | Logging utilities |
| alignment/model_utils.py | Model/tokenizer helper utilities |
| dataset_loader/mix_alpaca.py | SFT dataset loader and preprocessing (e.g., Alpaca mixes) |
| dataset_loader/sub_SHP_2.py | Preference dataset loader (SHP subset) |
| dataset_loader/sub_yahoo_answers_topics.py | SFT dataset loader (Yahoo Answers topics subset) |
| dataset_loader/ultrafeedback_binarized_enhancement.py | Preference dataset loader (UltraFeedback binarized) |
| evaluator/azure_gpt_client.py | Azure GPT client for evaluation |
| evaluator/eval_prompt.txt | Evaluation prompt template |
| evaluator/evaluate.py | Distributed generation and win/tie/loss evaluation |
| recipes/accelerate_configs/zero3.yaml | Accelerate config for DeepSpeed ZeRO-3 |
| recipes/*.yaml | Training recipes for SFT/DPO/SimPO (weighted and non-weighted) |
| scripts/generate_weight.py | Offline weight generation script |
| scripts/test.py | Distributed evaluation script |
| scripts/weighted_sft.py | Weighted SFT training entrypoint |
| scripts/weighted_dpo.py | Weighted DPO training entrypoint |
| scripts/weighted_sft_trainer.py | Weighted SFT Trainer implementation |
| scripts/weighted_dpo_trainer.py | Weighted DPO Trainer implementation |
| weight_function/uniform_weight.py | Uniform weight baseline |
| weight_function/weight_by_holdout.py | Example weight function using a holdout set |

## Data Formats

- SFT data (after preprocessing):
  - Each example:
    - message: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
    - text: string (flattened from messages via chat template)
    - example_id: int (stable ID for weighting)

- Preference data:
  - Fields:
    - chosen_messages: list of messages for the chosen answer
    - rejected_message: list of messages for the rejected answer
    - prompt: string (flattened prompt text)
    - chosen: string (chosen answer)
    - rejected: string (rejected answer)
    - example_id: int

Example preference entry:
- {"chosen_messages": [...], "rejected_message": [...], "prompt": "...", "chosen": "...", "rejected": "...", "example_id": 123}

## Installation

1) Create and activate the Conda environment:
```
conda env create -f environment.yaml
conda activate <your-env-name>
```

2) (Optional) Configure any credentials needed for evaluation (Azure OpenAI), if you plan to run the evaluator.

## Usage

### 1. Generate Weights Offline
Compute and cache weights for your training set:
```
python scripts/generate_weight.py --config recipes/<your_recipe>.yaml
```
This will use the configured weight_function (e.g., uniform_weight or holdout-based) and save weights to the specified cache file.

### 2. Train SFT
Run weighted SFT with DeepSpeed ZeRO-3 acceleration:
```
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml \
  scripts/weighted_sft.py \
  --config recipes/<your_recipe>.yaml \
  --output_dir <OUTPUT_DIR>
```

### 3. Train DPO
```
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml \
  scripts/weighted_dpo.py \
  --config recipes/<your_recipe>.yaml \
  --output_dir <OUTPUT_DIR>
```

### 4. Train SimPO
```
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml \
  scripts/simpo.py \
  --config recipes/<your_recipe>.yaml \
  --output_dir <OUTPUT_DIR>
```

### 5. Evaluate
Distributed generation and evaluation:
```
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/test.py \
  --config recipes/<your_recipe>.yaml \
  --output_dir <OUTPUT_DIR>
```

### 6. One-click Pipeline
Alternatively, you can run the entire pipeline via:
```
bash run.sh
```

## Notes and Tips
- The project infers the model’s response template automatically and uses a custom EoTCompletionCollator to include the <|eot_id|> token in labels where appropriate (necessary for TRL 0.9.6 which lacks label_eos_token=True).
- example_id is preserved through the data pipeline and carried in batches, ensuring correct alignment with per-example weights even under shuffling and distributed training.
- The provided zero3.yaml accelerates training with DeepSpeed ZeRO-3; adjust num_processes for your GPU count.
- Ensure your tokenizer’s chat_template aligns with your model (e.g., Llama-3-Instruct). The code falls back to a provided template if missing.

## Roadmap
1) Finalize and test SimPO
2) Unify and simplify run scripts
3) Add additional weight computation methods (e.g., RHO-Loss, One-shot learning)