import time
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

    def solve(self, problem: str, max_steps: int = 50) -> str:
        """
        Main loop.
        """
        print(f"\n[Solver] Problem: {problem[:50]}...", flush=True)
        
        # Init Client States
        # Should correspond to empty history
        # We can init them to 0 manually, or ask engine to init for empty string
        # Manual 0 is fine for start.
        client_states = [{'nll_sum': 0.0, 'token_len': 0} for _ in self.engine.clients]
        
        history_parts = []
        
        for step_idx in range(max_steps):
            print(f"\n--- Step {step_idx + 1} ---", flush=True)
            
            # Strategy Decision
            best_step, new_states = self.strategy.select_next_step(
                problem,
                history_parts,
                self.engine,
                client_states
            )
            
            if not best_step:
                print("Strategy returned no step. Stopping.")
                break
                
            history_parts.append(best_step)
            client_states = new_states
            
            # Termination Check
            if "\\boxed{" in best_step:
                print("Termination condition (boxed) met.", flush=True)
                break
        
        return "\n\n".join(history_parts)
