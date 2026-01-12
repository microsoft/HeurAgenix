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
        
        for step_idx in range(max_steps):
            # Safety Check: Prevent OOM from infinite loops
            # 25000 chars is roughly 6000-8000 tokens.
            current_context_len = len(problem) + sum(len(p) for p in history_parts)
            if current_context_len > 25000:
                logging.warning(f"Context length {current_context_len} exceeds safety limit (25000). Terminating early.")
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
                
            history_parts.append(best_step)
            client_states = new_states
            
            # Termination Check
            if "\\boxed{" in best_step:
                logging.info("Termination condition (boxed) met.")
                break
        logging.info(f"\n--------------------------Problem solved--------------------------\n")
        return "\n\n".join(history_parts)
