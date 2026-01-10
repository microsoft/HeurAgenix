import numpy as np
import logging
from typing import List, Dict, Tuple
from src.engine.strategy.base_strategy import BaseStrategy
from src.engine.engine import SwarmEngine

class VotingStrategy(BaseStrategy):
    """
    Outcome: One-Step Voting Consensus (formerly Greedy).
    Generates N candidates -> Scores N*N -> Selects Best Mean Score.
    """
    def __init__(self, aggregation: str = "mean", exclude_self: bool = False):
        self.aggregation = aggregation
        self.exclude_self = exclude_self

    def select_next_step(
        self, 
        problem: str, 
        history_parts: List[str], 
        engine: SwarmEngine,
        client_states: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        
        current_cot_text = "\n\n".join(history_parts)
        
        if engine.system_prompt:
            base_messages = [{"role": "system", "content": engine.system_prompt}, {"role": "user", "content": problem}]
        else:
            base_messages = [{"role": "user", "content": problem}]

        # 1. Generate Candidates (Parallel)
        step_candidates = engine.generate_candidates(problem, current_cot_text)
        
        if not step_candidates:
            # Fallback? Return empty?
            # Or raise error to abort.
            return "", client_states

        # 2. Score Candidates (Parallel)
        # scores_matrix[cand_idx][reviewer_idx]
        scores_matrix = engine.score_candidates(
            base_messages, 
            history_parts, 
            step_candidates, 
            client_states
        )

        # 3. Aggregate
        final_scores = []
        for c_idx, _ in enumerate(step_candidates):
            row_scores = scores_matrix[c_idx]
            valid_scores = [s for s in row_scores if s is not None]
            
            # TODO: Implement exclude_self logic here if tracking which client gen'd which cand
            # Since generate_candidates returns list ordered by clients, 
            # step_candidates[i] is from client[i].
            # So if exclude_self is True, we ignore row_scores[c_idx] (if it exists).
            
            if self.exclude_self and len(self.clients) > 1:
                # Strategy doesn't hold 'clients' directly, Engine does.
                # Assuming index alignment.
                if len(valid_scores) > 1: 
                     # Only remove if we have enough other scores
                     # Ideally we remove row_scores[c_idx]
                     pass
            
            if not valid_scores:
                final_scores.append(float('inf'))
            else:
                if self.aggregation == "mean":
                    final_scores.append(np.mean(valid_scores))
                elif self.aggregation == "min":
                    final_scores.append(np.min(valid_scores))
                else: 
                     final_scores.append(np.mean(valid_scores))

        # 4. Select
        if not final_scores or min(final_scores) == float('inf'):
            logging.info("No valid scores, picking first.")
            best_step = step_candidates[0]
            best_idx = 0
        else:
            best_idx = np.argmin(final_scores)
            best_step = step_candidates[best_idx]
        
        logging.info("Candidates:")
        for i, cand in enumerate(step_candidates):
            preview = cand.replace('\\n', ' ')
            marker = "*" if i == best_idx else " "
            sc = final_scores[i] if i < len(final_scores) else -1
            logging.info(f"  [avg_nll={sc:.4f}, {marker}] {preview}")

        # 5. Update States for the *chosen* path
        new_history = history_parts + [best_step]
        new_history_text = "\n\n".join(new_history)
        new_client_states = engine.update_states(base_messages, new_history_text)
        
        return best_step, new_client_states
