import logging
from typing import List, Dict, Tuple
from src.engine.strategy.base_strategy import BaseStrategy
from src.engine.engine import SwarmEngine

class SingleStrategy(BaseStrategy):
    """
    Outcome: Standard Chain-of-Thought (Step-by-Step).
    Logic: Generate next step. Accept it immediately. No scoring. No voting.
    Used for: Baseline comparison (Speed & Accuracy).
    """
    
    def select_next_step(
        self, 
        problem: str, 
        history_parts: List[str], 
        engine: SwarmEngine,
        client_states: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        
        current_cot_text = "\n\n".join(history_parts)
        
        # base_messages removed as it is unused in SingleStrategy (no NLL calc)

        # --- Dynamic Token Budget Calculation ---
        # Get limits from config (or use safe defaults for A100-40G)
        # hard_limit: Absolute Red Line (approx 16k tokens / 64000 chars)
        # soft_limit: Generation Warning Line (approx 12k tokens / 48000 chars)
        hard_limit_chars = engine.config.get('max_context_chars_hard', 64000)
        soft_limit_chars = engine.config.get('max_context_chars_soft', 48000)
        
        # Estimate current context length (approximated by chars)
        # Includes System Prompt, Problem, and History. System prompt length is roughly constant/small, ignoring for simplified estimation.
        current_len_chars = len(problem) + len(current_cot_text)
        
        # 1. Hard Limit Check (Circuit Breaker)
        if current_len_chars > hard_limit_chars:
            logging.warning(f"Strategy Hard Limit Reached: {current_len_chars} > {hard_limit_chars}. Terminating to prevent OOM.")
            return "", client_states

        # 2. Soft Limit Budgeting
        # Calculate remaining budget
        # We assume 1 token approx 3 chars (conservative). 
        # But here we work in CHARS for the threshold, and convert to TOKENS for the generation parameter.
        remaining_chars = soft_limit_chars - current_len_chars
        if remaining_chars <= 0:
            logging.warning(f"Strategy Soft Limit Reached. Forcing generation stop.")
            # We allow 1 token just to let it try to finish or output EOS, but effectively stopping.
            max_new_tokens_budget = 1
        else:
            # Convert chars to tokens usually div by 4, but div by 3 is safer buffer.
            # Ensure we don't exceed the standard single-step limit (e.g. 1024) even if we have budget.
            # Using 2.5 chars/token estimate to be extra safe for "tokens" budget.
            budget_tokens = int(remaining_chars / 2.5)
            max_new_tokens_budget = min(1024, budget_tokens) # Default step limit is still 1024
            
        if max_new_tokens_budget < 10:
             logging.warning(f"Low token budget remaining: {max_new_tokens_budget}. Finishing up.")

        # 1. Generate Candidates
        # Pass the dynamic budget
        step_candidates = engine.generate_candidates(problem, current_cot_text, max_new_tokens=max_new_tokens_budget)
        
        if not step_candidates:
            return "", client_states

        # Simple Logic: Pick the candidate from the first client.
        best_step = step_candidates[0]
        
        # Debug Log
        preview = best_step.replace('\\n', ' ')
        logging.info(f"{preview}")

        # 2. No Score Update for Single Strategy
        # To save memory (avoiding 9GB+ Logits Matrix allocation), we SKIP the NLL calculation.
        # Single strategy corresponds to 'Greedy Decoding' baseline effectively, which doesn't use the score.
        # We just return the old states (or dummy) to keep interface compatible.
        
        new_history = history_parts + [best_step]
        # new_history_text = "\n\n".join(new_history)
        # new_client_states = engine.update_states(base_messages, new_history_text)
        
        return best_step, client_states
