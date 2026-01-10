import os
import json
import argparse
import sys
import time
from tqdm import tqdm
from typing import List, Type

# NEW Architecture Imports
from src.engine.engine import SwarmEngine
from src.engine.solver import ValidatingSolver
from src.engine.strategy.voting_strategy import VotingStrategy
from src.engine.strategy.consensus_value_strategy import ConsensusValueStrategy
from src.engine.strategy.single_strategy import SingleStrategy

from src.tasks.math500_task import Math500Task
from src.tasks.base_task import BaseTask

# Registry for available tasks
TASK_REGISTRY = {
    "math500": Math500Task
}

def run_consensus_evaluation(
    model_config_paths: List[str],
    task_name: str,
    strategy_name: str = "single",
    subset: str = "test",
    exp_name: str = ""
):
    # 1. Setup
    if not exp_name:
        exp_name = time.strftime("%Y%m%d_%H%M%S")
    
    base_output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..", "..", "ccdm", "output") if os.getenv("AMLT_OUTPUT_DIR") else "output"
    output_dir = os.path.join(base_output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Consensus Evaluation ---", flush=True)
    print(f"Model Configs: {model_config_paths}", flush=True)
    print(f"Task: {task_name} ({subset})", flush=True)
    print(f"Strategy: {strategy_name}", flush=True)
    print(f"Output Directory: {output_dir}", flush=True)

    task_class: Type[BaseTask] = TASK_REGISTRY[task_name]
    task = task_class(subset=subset)
    dataset = task.get_dataset()

    # Initialize Engine (Layer 2)
    engine = SwarmEngine(model_config_paths, system_prompt=task.system_prompt)
    
    # Select Strategy (Layer 3)
    if strategy_name == "voting":
        strategy = VotingStrategy()
    elif strategy_name == "consensus_value":
        strategy = ConsensusValueStrategy() # Using default mean aggregation
    elif strategy_name == "single":
        strategy = SingleStrategy()
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Initialize Solver (Layer 4)
    solver = ValidatingSolver(engine, strategy)

    # 2. Evaluation Loop
    correct_count = 0
    total_count = 0
    results = []
    pbar = tqdm(dataset)

    for item in pbar:
        # Format Prompt
        
        problem = item["problem"]
        ground_truth = item["ground_truth"]
        
        # Solver Execution (Layer 4)
        start_time = time.time()
        best_response = solver.solve(problem, max_steps=50) 
        elapsed = time.time() - start_time
        
        # Extract & Verify
        prediction = task.extract_answer(best_response)
        
        is_correct = task.verify_answer(prediction, ground_truth)
        
        if is_correct:
            correct_count += 1
        total_count += 1
        
        # Log Result
        result_entry = {
            "system_prompt": task.system_prompt,
            "problem": problem,
            "ground_truth": ground_truth,
            "response": best_response,
            "prediction": prediction,
            "is_correct": is_correct,
            "time_taken": elapsed
        }
        results.append(result_entry)
        
        # Update progress bar
        current_acc = (correct_count / total_count) * 100
        pbar.set_description(f"Acc: {current_acc:.2f}% ({correct_count}/{total_count})")

    # 3. Summary & Save
    final_acc = (correct_count / total_count) * 100
    print(f"\n--- Evaluation Complete ---", flush=True)
    print(f"Final Accuracy: {final_acc:.2f}%", flush=True)
    
    # Get configs from engine clients for logging
    configs = [client.config for client in engine.clients]
    
    # Generate filenames
    metrics_file = os.path.join(output_dir, "metrics.json")
    generations_file = os.path.join(output_dir, "generations.json")
    
    generation_details = []
    for item in results:
        generation_details.append({
            "system_prompt": item["system_prompt"],
            "problem": item["problem"],
            "response": item["response"],
            "ground_truth": item["ground_truth"]
        })

    with open(metrics_file, 'w') as f:
        json.dump({
            "configs": configs,
            "task": task_name,
            "strategy": strategy_name,
            "accuracy": final_acc,
            "total": total_count,
            "details": results
        }, f, indent=2)
        
    with open(generations_file, 'w') as f:
        json.dump({
            "configs": configs,
            "task": task_name,
            "strategy": strategy_name,
            "details": generation_details
        }, f, indent=2)
    
    print(f"Generations saved to {generations_file}", flush=True)
    print(f"Metrics saved to {metrics_file}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run consensus evaluation on a task.")
    # Allow multiple config files
    parser.add_argument("-c", "--configs", type=str, required=True, help="Paths to LLM config jsons (comma separated)")
    parser.add_argument("-t", "--task", type=str, default="math500", help="Task name")
    parser.add_argument("-s", "--strategy", type=str, default="single", choices=["single", "voting", "consensus_value"], help="Strategy name")
    parser.add_argument("-e", "--exp_name", type=str, default="", help="Experiment name (default: timestamp)")

    args = parser.parse_args()
    
    run_consensus_evaluation(
        args.configs.split(","),
        args.task,
        strategy_name=args.strategy,
        exp_name=args.exp_name
    )
