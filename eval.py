import os
import json
import argparse
import sys
import time
import logging
from tqdm import tqdm
from typing import List, Type
from src.engine.engine import SwarmEngine
from src.engine.solver import ValidatingSolver
from src.engine.strategy.voting_strategy import VotingStrategy
from src.engine.strategy.consensus_value_strategy import ConsensusValueStrategy
from src.engine.strategy.single_strategy import SingleStrategy
from src.tasks.math500_task import Math500Task
from src.tasks.base_task import BaseTask

# Custom handler for BlobFuse synchronization
class DirectFileHandler(logging.Handler):
    def __init__(self, filename, mode='a', encoding='utf-8'):
        super().__init__()
        self.filename = filename
        self.mode = mode
        self.encoding = encoding
        
        # Initialize file (truncate if mode is 'w')
        if mode == 'w':
             with open(self.filename, 'w', encoding=self.encoding) as f:
                pass
             self.mode = 'a' # Switch to append for subsequent writes

    def emit(self, record):
        try:
            msg = self.format(record)
            # Force Open-Write-Close for every log to ensure BlobFuse sync
            with open(self.filename, self.mode, encoding=self.encoding) as f:
                f.write(msg + '\n')
        except Exception:
            self.handleError(record)

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
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if not exp_name:
        exp_name = timestamp
    else:
        # Append timestamp to user-provided name to prevent overwrites
        exp_name = f"{exp_name}_{timestamp}"
    
    base_output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..", "..", "ccdm", "output") if os.getenv("AMLT_OUTPUT_DIR") else "output"
    output_dir = os.path.join(base_output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup Logging
    log_file = os.path.join(output_dir, "run.log")
    
    # Remove existing handlers if any (to avoid duplicate logs if run multiple times in same session)
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            DirectFileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"--- Evaluation ---")
    logger.info(f"Model Configs: {model_config_paths}")
    logger.info(f"Task: {task_name} ({subset})")
    logger.info(f"Strategy: {strategy_name}")
    logger.info(f"Output Directory: {output_dir}")

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

    # Pre-calculate Paths
    # Consolidated results file
    results_file = os.path.join(output_dir, "results.json")

    def save_results():
        # Get configs from engine clients for logging
        # We access this lazily as engine is init before loop
        configs = [client.config for client in engine.clients]
        
        current_acc = (correct_count / total_count) * 100 if total_count > 0 else 0.0
        
        # Consolidate everything into one JSON
        with open(results_file, 'w') as f:
            json.dump({
                "meta": {
                    "task": task_name,
                    "strategy": strategy_name,
                    "model_configs": configs # Full config content
                },
                "stats": {
                    "accuracy": current_acc,
                    "correct_count": correct_count,
                    "total_count": total_count,
                    "total_problems": len(dataset),
                    "processed_problems": total_count,
                },
                "results": results # Full details including prompts, responses, pred, is_correct
            }, f, indent=2)

    save_results()
    for item in pbar:
        # Format Prompt
        
        problem = item["problem"]
        ground_truth = item["ground_truth"]
        
        # Solver Execution (Layer 4)
        start_time = time.time()
        best_response = solver.solve(problem, max_steps=50, problem_index=total_count + 1) 
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
        
        # Checkpoint every item for real-time updates
        save_results()
        
    # 3. Summary & Save
    final_acc = (correct_count / total_count) * 100
    logger.info(f"\n--- Evaluation Complete ---")
    logger.info(f"Final Accuracy: {final_acc:.2f}%")
    
    save_results()
    
    logger.info(f"Results saved to {results_file}")
    logger.info(f"Log saved to {log_file}")

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
