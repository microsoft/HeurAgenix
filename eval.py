import os
# Fix fragmentation issues for OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import yaml
import shutil
import argparse
import sys
import time
import logging
import torch
from tqdm import tqdm
from typing import List, Type, Dict
from src.engine.engine import SwarmEngine
from src.engine.solver import ValidatingSolver
from src.engine.strategy.voting_strategy import VotingStrategy
from src.engine.strategy.consensus_value_strategy import ConsensusValueStrategy
from src.engine.strategy.single_strategy import SingleStrategy
from src.tasks.math500_task import Math500Task
from src.tasks.aime_task import AimeTask
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
    "math500": Math500Task,
    "aime": AimeTask
}

def run_consensus_evaluation(
    config_path: str,
    exp_name: str = None
):
    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Determine Experiment Name & Output Dir
    # Priority: CLI Override > Config['exp_name'] > Default
    exp_name = exp_name if exp_name else config.get('exp_name', f"experiment_{int(time.time())}")
    
    # Update config with final exp_name for logging
    config['exp_name'] = exp_name
    
    base_output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..", "..", "ccdm", "output") if os.getenv("AMLT_OUTPUT_DIR") else "output"
    output_dir = os.path.join(base_output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy config file to output dir for reproducibility
    shutil.copy(config_path, os.path.join(output_dir, "config.yaml"))
    
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
    
    # Extract Params
    task_name = config.get('task', {}).get('name', 'math500')
    strategy_config = config.get('strategy', {})
    strategy_name = strategy_config.get('type', 'single')
    models_config = config.get('models', [])
    engine_config = config.get('engine', {})
    
    logger.info(f"--- Evaluation ---")
    logger.info(f"Config File: {config_path}")
    logger.info(f"Exp Name: {exp_name}")
    logger.info(f"Task: {task_name}")
    logger.info(f"Strategy: {strategy_name}")
    logger.info(f"Output Directory: {output_dir}")

    try:
        task_class: Type[BaseTask] = TASK_REGISTRY[task_name]
        
        # Instantiate task with subset/subset config if applicable
        task_specific_config = config.get('task', {})
        subset = task_specific_config.get('subset')
        
        if subset:
            task = task_class(subset=subset)
        else:
            task = task_class()

        # Load Task Data
        test_data = task.get_dataset()

        logger.info(f"Total problems to evaluate: {len(test_data)}")

        # Initialize Engine (Layer 2)
        # Pass model list dicts directly
        engine = SwarmEngine(models_config, system_prompt=task.system_prompt, config=engine_config)
        
        # Select Strategy (Layer 3)
        agg_method = strategy_config.get('aggregation', 'mean')
        exclude_self = strategy_config.get('exclude_self', False)

        if strategy_name == "voting":
            strategy = VotingStrategy(aggregation=agg_method, exclude_self=exclude_self)
        elif strategy_name == "consensus_value":
            strategy = ConsensusValueStrategy(aggregation=agg_method, exclude_self=exclude_self) 
        elif strategy_name == "single":
            strategy = SingleStrategy()
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Initialize Solver (Layer 4)
        # Pass the full engine config directly.
        # The Solver will extract 'max_steps' and other relevant parameters.
        solver = ValidatingSolver(engine, strategy=strategy, config=engine_config)

        # 2. Evaluation Loop
        correct_count = 0
        total_count = 0
        results = []
        pbar = tqdm(test_data)

        # Pre-calculate Paths
        # Consolidated results file
        results_file = os.path.join(output_dir, "results.json")

        def save_results():
            # Get configs from engine clients for logging
            # We access this lazily as engine is init before loop
            
            current_acc = (correct_count / total_count) * 100 if total_count > 0 else 0.0
            
            # Consolidate everything into one JSON
            with open(results_file, 'w') as f:
                json.dump({
                    "accuracy": current_acc,
                    "correct_count": correct_count,
                    "total_count": total_count,
                    "total_problems": len(test_data),
                    "processed_problems": total_count,
                    "results": results 
                }, f, indent=2)

        save_results()

        for item in pbar:
            # Reset memory stats
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            problem = item["problem"]
            ground_truth = item["ground_truth"]
            
            # Solver Execution (Layer 4)
            start_time = time.time()
            # Max steps is handled by solver internally using config now
            best_response = solver.solve(problem, problem_index=total_count + 1) 
            elapsed = time.time() - start_time
            
            # Record Memory
            max_memory_gb = 0.0
            if torch.cuda.is_available():
                max_memory_bytes = torch.cuda.max_memory_allocated()
                max_memory_gb = max_memory_bytes / (1024 ** 3)
            
            # Extract & Verify
            prediction = task.extract_answer(best_response)
            
            is_correct = task.verify_answer(prediction, ground_truth)
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # Log Result
            logger.info(f"Problem {total_count} | Peak Memory: {max_memory_gb:.2f} GB | Time: {elapsed:.2f}s")
            
            result_entry = {
                "system_prompt": task.system_prompt,
                "problem": problem,
                "ground_truth": ground_truth,
                "response": best_response,
                "prediction": prediction,
                "is_correct": is_correct,
                "time_taken": elapsed,
                "peak_memory_gb": max_memory_gb
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

    except Exception:
        logger.critical("Fatal error occurred during evaluation:", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run consensus evaluation via YAML config.")
    
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("-e", "--exp_name", type=str, default=None, help="Override Experiment name")

    args = parser.parse_args()
    
    run_consensus_evaluation(
        config_path=args.config,
        exp_name=args.exp_name
    )
