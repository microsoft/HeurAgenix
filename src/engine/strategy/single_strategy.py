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
        # Switch to Token-based limit for precision and VRAM safety
        hard_limit_tokens = engine.config.get('max_context_tokens_hard', 16000)
        soft_limit_tokens = engine.config.get('max_context_tokens_soft', 12000)

        # Estimate usage properly using tokenizer from first available client
        if engine.clients and hasattr(engine.clients[0], 'pipeline'):
            tokenizer = engine.clients[0].pipeline.tokenizer
            
            # Reconstruct message structure for accurate token counting
            if engine.system_prompt:
                messages = [{"role": "system", "content": engine.system_prompt}, {"role": "user", "content": problem}]
            else:
                messages = [{"role": "user", "content": problem}]
            
            if current_cot_text:
                messages.append({"role": "assistant", "content": current_cot_text})
                
            # Apply template to get real prompt length
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False)
            tokenized_ids = tokenizer(prompt_str, return_tensors='pt')['input_ids']
            current_len_tokens = tokenized_ids.shape[1]
        else:
            # Fallback estimation if tokenizer not accessible (e.g. API client)
            current_len_tokens = (len(problem) + len(current_cot_text)) // 3
            logging.warning("Tokenizer not found, using char/3 estimation.")
        
        # 1. Hard Limit Check (Circuit Breaker)
        if current_len_tokens > hard_limit_tokens:
            logging.warning(f"Strategy Hard Limit Reached: {current_len_tokens} > {hard_limit_tokens} tokens. Terminating to prevent OOM.")
            return "", client_states

        # 2. Soft Limit Budgeting
        remaining_tokens = soft_limit_tokens - current_len_tokens
        if remaining_tokens <= 0:
            logging.warning(f"Strategy Soft Limit Reached. Forcing generation stop.")
            # We allow 1 token just to let it try to finish or output EOS, but effectively stopping.
            max_new_tokens_budget = 1
        else:
            # Using token budget directly.
            # Ensure we don't exceed the standard single-step limit (e.g. 1024) even if we have budget.
            max_new_tokens_budget = min(1024, int(remaining_tokens))
            
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
