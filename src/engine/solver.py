import time
import logging
from typing import List, Type
from src.engine.engine import SwarmEngine
from src.engine.strategy.base_strategy import BaseStrategy
from src.engine.strategy.voting_strategy import VotingStrategy

class ValidatingSolver:
    """
    Layer 4: Solver
    Orchestrates the solution process using Engine and Strategy.
    """
    def __init__(self, engine: SwarmEngine, strategy: BaseStrategy = None):
        self.engine = engine
        self.strategy = strategy if strategy else VotingStrategy()

    def solve(self, problem: str, max_steps: int = 50, problem_index: int = None) -> str:
        """
        Main loop.
        """
        if problem_index:
            problem_str = f"\n--------------------------Problem index[{problem_index}]--------------------------\n"
        else:
            problem_str = "\n--------------------------Problem--------------------------\n"
        problem_str += f"\nProblem: {problem}\n"
        logging.info(problem_str)
        
        # Init Client States
        # Should correspond to empty history
        # We can init them to 0 manually, or ask engine to init for empty string
        # Manual 0 is fine for start.
        client_states = [{'nll_sum': 0.0, 'token_len': 0} for _ in self.engine.clients]
        
        history_parts = []
        
        # OOM Prevention: Cap max steps aggressively for baseline runs
        run_max_steps = min(max_steps, 25)

        for step_idx in range(run_max_steps):
            # Safety Check: Prevent OOM from infinite loops
            # Lower limit to ~8000 chars (approx 2000 tokens) to ensure dual-model safety on single GPU
            current_context_len = len(problem) + sum(len(p) for p in history_parts)
            if current_context_len > 12000:
                logging.warning(f"Context length {current_context_len} exceeds safety limit (12000). Terminating early.")
                break

            logging.info(f"--- Step {step_idx + 1} ---")
            
            # Strategy Decision
            best_step, new_states = self.strategy.select_next_step(
                problem,
                history_parts,
                self.engine,
                client_states
            )
            
            if not best_step:
                logging.info("Strategy returned no step. Stopping.")
                break

            # Loop Detection: Simple duplication check
            if history_parts and best_step.strip() == history_parts[-1].strip():
                logging.warning("Detected exact repetition of previous step. Terminating early to prevent loop.")
                break
            
            # Heuristic Loop Detection: Check if generating extremely similar content (e.g. Case 101, Case 102...)
            # If the step is very short and we are deep in steps, risk is high.
            if step_idx > 10 and len(best_step) < 200:
                # Calculate Jaccard Set similarity or simple substring overlap?
                pass 
                
            history_parts.append(best_step)
            client_states = new_states
            
            # Termination Check
            if "\\boxed{" in best_step:
                logging.info("Termination condition (boxed) met.")
                break
        logging.info(f"\n--------------------------Problem solved--------------------------\n")
        return "\n\n".join(history_parts)
