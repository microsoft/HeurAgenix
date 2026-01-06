import os
import json
import argparse
import sys
import time
from tqdm import tqdm
from typing import List, Dict, Type

# Ensure src is in python path
sys.path.append(os.getcwd())

from src.engine.consensus_engine import ConsensusEngine
from src.tasks.math500_task import Math500Task
from src.tasks.base_task import BaseTask

# Registry for available tasks
TASK_REGISTRY = {
    "math500": Math500Task
}

def load_model_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return json.load(f)

def run_consensus_evaluation(
    model_config_paths: List[str],
    task_name: str,
    subset: str = "test",
    output_dir: str = "results_consensus"
):
    # 1. Setup
    print(f"--- Consensus Evaluation ---")
    print(f"Model Configs: {model_config_paths}")
    print(f"Task: {task_name} ({subset})")

    # Load Configs
    configs = []
    for path in model_config_paths:
        try:
            cfg = load_model_config(path)
            configs.append(cfg)
        except Exception as e:
            print(f"Failed to load config {path}: {e}")
            return

    # Initialize Engine
    try:
        # Pass configs to the engine, it will instantiate clients
        engine = ConsensusEngine(configs, output_dir=os.path.join(output_dir, "logs"))
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return

    # Load Task
    if task_name not in TASK_REGISTRY:
        print(f"Error: Task '{task_name}' not found. Available: {list(TASK_REGISTRY.keys())}")
        return
    
    task_class: Type[BaseTask] = TASK_REGISTRY[task_name]
    task = task_class(subset=subset)
    dataset = task.get_dataset()

    # 2. Evaluation Loop
    correct_count = 0
    total_count = 0
    results = []

    pbar = tqdm(dataset)
    for i, item in enumerate(pbar):
        # Format Prompt
        messages = task.format_prompt(item)
        
        # Engine Decision
        # The engine is responsible for coordinating multiple models
        try:
            start_time = time.time()
            best_response = engine.decide(messages) 
            elapsed = time.time() - start_time
        except Exception as e:
            print(f"\nError processing sample {i}: {e}")
            best_response = ""
            elapsed = 0
        
        # Extract & Verify
        prediction = task.extract_answer(best_response)
        ground_truth = item["ground_truth"]
        is_correct = task.verify_answer(prediction, ground_truth)
        
        if is_correct:
            correct_count += 1
        total_count += 1
        
        # Record result
        result_entry = {
            "problem": item["problem"],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "response": best_response,
            "is_correct": is_correct,
            "time_taken": elapsed
        }
        results.append(result_entry)
        
        # Update progress bar
        current_acc = (correct_count / total_count) * 100
        pbar.set_description(f"Acc: {current_acc:.2f}% ({correct_count}/{total_count})")

    # 3. Summary & Save
    final_acc = (correct_count / total_count) * 100
    print(f"\n--- Evaluation Complete ---")
    print(f"Final Accuracy: {final_acc:.2f}%")
    
    os.makedirs(output_dir, exist_ok=True)
    # Generate a run name based on number of models
    run_name = f"consensus_{len(configs)}models"
    result_file = os.path.join(output_dir, f"{run_name}_{task_name}_results.json")
    
    # Save a simplified version of configs to avoid clutter
    simple_configs = [{"name": c.get("name"), "model_path": c.get("model_path")} for c in configs]

    with open(result_file, 'w') as f:
        json.dump({
            "configs": simple_configs,
            "task": task_name,
            "accuracy": final_acc,
            "total": total_count,
            "details": results
        }, f, indent=2)
    
    print(f"Results saved to {result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run consensus evaluation on a task.")
    # Allow multiple config files
    parser.add_argument("-c", "--configs", type=str, nargs='+', required=True, help="Paths to LLM config jsons (space separated)")
    parser.add_argument("-t", "--task", type=str, default="math500", help="Task name")
    parser.add_argument("-o", "--output", type=str, default="results_consensus", help="Output directory")
    
    args = parser.parse_args()
    
    run_consensus_evaluation(
        args.configs,
        args.task,
        output_dir=args.output
    )
