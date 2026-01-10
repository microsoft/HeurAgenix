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
        
        if engine.system_prompt:
             base_messages = [{"role": "system", "content": engine.system_prompt}, {"role": "user", "content": problem}]
        else:
             base_messages = [{"role": "user", "content": problem}]

        # 1. Generate Candidates
        # Even if we have N models, SingleStrategy usually implies we just check ONE model.
        # But if the user passed N configs, maybe they want N independent chains?
        # For 'consensus' framework, if we use SingleStrategy with N models, 
        # it's ambiguous. 
        # Assumption: We only use the FIRST client to generate.
        
        # To reuse engine's parallel structure but only use Client 0:
        # We can just call generate and pick the first one.
        
        step_candidates = engine.generate_candidates(problem, current_cot_text)
        
        if not step_candidates:
            return "", client_states

        # Simple Logic: Pick the candidate from the first client.
        best_step = step_candidates[0]
        
        # Debug Log
        preview = best_step.replace('\\n', ' ')
        logging.info(f"  [Single] {preview}")

        # 2. Update States
        # We still need to update states to keep the 'token_len' correct 
        # if we were to mix strategies, or just to be consistent.
        # But specifically for SingleStrategy, we don't use the NLL state.
        # However, to avoid errors if we switch strategies mid-stream (future feature),
        # we perform the update.
        
        new_history = history_parts + [best_step]
        new_history_text = "\n\n".join(new_history)
        new_client_states = engine.update_states(base_messages, new_history_text)
        
        return best_step, new_client_states
