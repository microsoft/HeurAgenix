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

def run_consensus_evaluation(
    model_config_paths: List[str],
    task_name: str,
    subset: str = "test",
    exp_name: str = ""
):
    # 1. Setup
    if not exp_name:
        exp_name = time.strftime("%Y%m%d_%H%M%S")
    
    base_output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..", "..", "ccdm", "output") if os.getenv("AMLT_OUTPUT_DIR") else "output"
    output_dir = os.path.join(base_output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Consensus Evaluation ---")
    print(f"Model Configs: {model_config_paths}")
    print(f"Task: {task_name} ({subset})")
    print(f"Output Directory: {output_dir}")

    task_class: Type[BaseTask] = TASK_REGISTRY[task_name]
    task = task_class(subset=subset)
    dataset = task.get_dataset()

    # Initialize Engine
    engine = ConsensusEngine(model_config_paths, system_prompt=task.system_prompt)


    # 2. Evaluation Loop
    correct_count = 0
    total_count = 0
    results = []
    pbar = tqdm(dataset)

    for item in pbar:
        # Format Prompt
        
        problem = item["problem"]
        ground_truth = item["ground_truth"]
        
        # Engine Decision
        # The engine is responsible for coordinating multiple models
        start_time = time.time()
        best_response = engine.decide(problem) 
        elapsed = time.time() - start_time
        
        # Extract & Verify
        prediction = task.extract_answer(best_response)
        
        is_correct = task.verify_answer(prediction, ground_truth)
        
        if is_correct:
            correct_count += 1
        total_count += 1
        
        # Log Result
        results.append({
            "system_prompt": task.system_prompt,
            "problem": problem,
            "ground_truth": ground_truth,
            "response": best_response,
            "prediction": prediction,
            "is_correct": is_correct,
            "time_taken": elapsed
        })
        
        pbar.set_description(f"Acc: {correct_count/total_count:.2%} ({correct_count}/{total_count})")
        
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
    
    # Get configs from engine clients for logging
    configs = [client.config for client in engine.clients]
    
    # Generate a run name based on number of models
    result_file = os.path.join(output_dir, f"results.json")
    
    # Generate a run name based on number of models - using len(configs)
    # result_file = os.path.join(output_dir, f"consensus_{len(configs)}models_{task_name}_results.json")

    with open(result_file, 'w') as f:
        json.dump({
            "configs": configs,
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
    parser.add_argument("-e", "--exp_name", type=str, default="", help="Experiment name (default: timestamp)")

    args = parser.parse_args()
    
    run_consensus_evaluation(
        args.configs,
        args.task,
        exp_name=args.exp_name
    )
