
# NaDRO: Leveraging Dual-Reward Strategies for LLMs Training on Noisy Data

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

 This repository contains the implementation for the paper: **"NaDRO: Leveraging Dual-Reward Strategies for LLMs Training on Noisy Data"**, which is based on an earlier version of the **HeurAgenix** repository.

## Abstract

Group Relative Policy Optimization (GRPO) fine-tuning has demonstrated significant enhancements in reasoning tasks. However, it often relies on high quality labeled dataset, which is typically difficult to obtain. To address this challenge, we introduce **N**oise-**A**ware **D**ual-**R**eward **O**ptimization (**NaDRO**) to effectively enhances the training of Large Language Models (LLMs) under noisy or ambiguous supervision. NaDRO operates through two key components: **(1) Preference-based Outcome Reward (POR)**,which makes a principled bias-variance tradeoff, reducing training variance by learning from robust preference rankings instead of overfitting to single-best estimates; and **(2) Context Perception Reward (CPR) mechanism**, which ensures that LLMs conduct necessary qualitative assessment of the current problem state to foster deeper situational understanding prior to decision-making. To validate our approach in a realistic decision-making testbed, we model classic combinatorial optimization problems like the Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP) as Markov Decision Processes, generating training data via cost-limited exploration. Our results demonstrate that the fine-tuned Qwen 7B and Llama 3-8B models achieve statistically robust performance, significantly outperforming leading LLM baselines and standard fine-tuning methods on these complex benchmarks.

## Repository Structure

The repository is organized as follows:

```

.
├── inference/                \# Scripts and utilities for running inference
│   └── launch_hyper_heuristic.py \# Main script for inference
│   └── find_best.py               \# TTS script
│   └── src/                    \# Inference source code (problems, heuristics, etc.)
├── training/                 \# Scripts and utilities for model training
│   └── train.py          \# Main script for NaDRO training (example name)
│   └── model.py                \# Model definition
│   └── dataset.py              \# Data loading and processing
│   └── rewards.py              \# NaDRO reward function implementations
│   └── config.py               \# Training configurations
│   └── utils.py                \# Utility functions
├── requirements.txt          \# Python package dependencies
└── README.md                 \# This file

````

## Installation

### Prerequisites
* Python 3.12
* Access to Large Language Models (either local or via API like Azure OpenAI)

### Setup Steps

1.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    You might also need to install PyTorch separately according to your CUDA version if you plan to use GPUs. Refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for instructions.
    Ensure Unsloth dependencies are correctly installed if using Unsloth for training.

## Training with NaDRO

The training process uses the NaDRO methodology to fine-tune Large Language Models.

1.  **Navigate to the training directory:**
    ```bash
    cd training
    ```

2.  **Configuration:**
    * Training parameters, including model specifics, dataset paths, and NaDRO reward configurations, are primarily managed in `config.py` via `TRAINING_ARGS`.
    * Model details are in `model.py`.
    * Dataset loading logic is in `dataset.py`.
    * The core NaDRO reward functions (POR and CPR components) are implemented in `rewards.py`.

3.  **Run the training script:**
    The main training script orchestrates the GRPO fine-tuning process with the NaDRO rewards.
    ```bash
    python train.py
    ```
    (Ensure that any necessary environment variables, such as `WANDB_API_KEY` for Weights & Biases logging, are set if `report_to=["wandb"]` is configured in `TRAINING_ARGS`.)


## Inference

The inference scripts allow you to use the fine-tuned models (or other LLMs) to dynamically select heuristics for combinatorial optimization problems.

1.  **Navigate to the inference directory:**
    ```bash
    cd inference
    ```

2.  **LLM Configuration:**
    Create a JSON configuration file (e.g., `llm_settings.json`) to specify your LLM API details or local model endpoint. This file will be passed to the inference script using the `-l` or `--llm_setting` argument.

    **Example `llm_settings.json`:**
    ```json
    {
      "api_type": "local", // Can be "local" or "azure"
      // --- Azure OpenAI Settings (only if api_type is "azure") ---
      "api_base": "YOUR_AZURE_OPENAI_ENDPOINT",     // e.g., [https://your-resource-name.openai.azure.com/](https://your-resource-name.openai.azure.com/)
      "api_version": "YOUR_API_VERSION",             // e.g., 2024-02-01 or similar
      "azure_endpoint": "YOUR_AZURE_OPENAI_ENDPOINT",// Usually same as api_base
      "model": "YOUR_AZURE_DEPLOYMENT_NAME",         // Your model deployment name on Azure
      // --- Local Model Settings (only if api_type is "local") ---
      "local_endpoint": "[http://127.0.0.1:8000/v1/chat/completions](http://127.0.0.1:8000/v1/chat/completions)", // Or your local LLM's chat/completion endpoint (e.g. for vLLM or TGI)
      "local_model": "path/to/your/finetuned_lora_model_or_base_model_identifier", // Identifier or path for your local model
      // --- Common Generation Parameters ---
      "temperature": 0.7,
      "top_p": 0.95,
      "max_tokens": 800,
      "max_attempts": 5,    // Max retry attempts for API calls
      "sleep_time": 10      // Sleep time in seconds between retries
    }
    ```
    **Note:** For `local_model` when `api_type` is `local`, this might refer to a model identifier that your local inference server (specified by `local_endpoint`) can understand, or a path if the server loads models directly by path. The fine-tuned LoRA adapters from the training step need to be merged with the base model or loaded appropriately by your local serving solution.

3.  **Run Inference:**
    Use the `launch_hyper_heuristic.py` script.

    **Arguments:**
    * `-p, --problem`: Type of problem to solve (e.g., `tsp`, `cvrp`). Choices are dynamically determined from subdirectories in `inference/src/problems/`.
    * `-e, --heuristic`: Name or path of the heuristic function.
        * `gpt_hh`: Use an LLM (configured via `--llm_setting`) to select heuristics.
        * `random_hh`: Use random heuristic selection.
        * `or_solver`: Use a dedicated OR solver for the problem (if implemented).
        * You can also provide a path to a specific heuristic module.
    * `-d, --heuristic_dir`: Directory containing heuristic functions. Defaults to a problem-specific path like `src/problems/{problem}/heuristics/basic_heuristics`.
    * `-c, --test_case`: Path for a single test case file.
    * `-t, --test_dir`: Directory for a whole test set (processes all files in it).
    * `-r, --dump_trajectory`: Whether to dump the solution trajectory.
    * `-o, --output_dir`: Directory name for saving outputs. Defaults to a generated name based on heuristic, model, and timestamp.
    * `-m, --CPR_mode`: Enable the Context Perception Reward (CPR) mode logic during inference with LLM-based heuristic selection. This flag indicates the LLM should generate outputs considering the "cards" format as used in CPR.
    * `-l, --llm_setting`: Path to the LLM settings JSON file (as described above). Required for `gpt_hh`.

    **Example Command (using a fine-tuned model locally):**
    ```bash
    # Ensure your local LLM server is running and configured in llm_settings.json
    # Set TOKENIZERS_PARALLELISM to false if you encounter warnings with Hugging Face tokenizers
    export TOKENIZERS_PARALLELISM=false

    python launch_hyper_heuristic.py \
        -p tsp \
        -e gpt_hh \
        -d src/problems/tsp/heuristics/basic_heuristics \
        -c src/problems/tsp/data/test_tsp/bier127.tsp \
        -m \
        -l llm_settings.json \
        -o results_bier127_nadro_tsp
    ```

