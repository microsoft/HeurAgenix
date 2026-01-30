import numpy as np
import logging
from typing import List, Dict, Tuple
from src.engine.strategy.base_strategy import BaseStrategy
from src.engine.engine import SwarmEngine

class ConsensusValueStrategy(BaseStrategy):
    """
    Outcome: Two-Step Consensus (State Value Estimation) (formerly Lookahead).
    Goal: Pick the step R_i that leads to the most 'promising' state S_i.
    
    Algorithm:
    1. Generate N candidates [R_1, ..., R_N].
    2. Form N hypothetical states [S_1, ..., S_N].
    3. For EACH S_i:
        a. Generate M next-steps [R'_i1, ..., R'_iM] (Lookahead).
        b. Score these M steps in context of S_i.
        c. Value V(S_i) = Aggregation of scores (e.g., mean of best agreement).
    4. Select R_i with best V(S_i).
    
    Complexity: N * M generations + N * M * N evaluations.
    If M=N (all agents predict), then O(N^3) evals.
    """
    def __init__(self, aggregation: str = "mean", exclude_self: bool = False, value_metric: str = "mean_nll", alpha: float = 1.0, info_weight: float = 0.5):
        self.aggregation = aggregation
        self.exclude_self = exclude_self
        # value_metric: "mean_nll" (default), "sum_nll", or "alpha_nll"
        self.value_metric = value_metric
        self.alpha = alpha
        self.info_weight = info_weight

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

        # --- Dynamic Token Budget Calculation (Layer 1) ---
        # Switch to Token-based limit for precision and VRAM safety
        hard_limit_tokens = engine.config.get('max_context_tokens_hard', 16000)
        soft_limit_tokens = engine.config.get('max_context_tokens_soft', 12000)
        
        # Estimate usage properly using tokenizer from first available client
        test_history = base_messages + [{"role": "assistant", "content": current_cot_text}]
        current_len_tokens = (len(problem) + len(current_cot_text)) // 3
        
        if engine.clients:
            try:
                current_len_tokens = engine.clients[0].compute_token_count(test_history)
            except Exception as e:
                # Fallback estimation if calculation fails
                pass

        if current_len_tokens > hard_limit_tokens:
            logging.warning(f"Strategy Hard Limit Reached: {current_len_tokens} tokens. Terminating.")
            return "", client_states
             
        remaining_tokens = soft_limit_tokens - current_len_tokens
        if remaining_tokens <= 0:
            max_new_tokens_budget = 1
        else:
            max_new_tokens_budget = min(1024, int(remaining_tokens))

        # --- Step 1: Broad Search (First Layer Generation) ---
        # Generate N candidates R_i
        layer1_candidates = engine.generate_candidates(problem, current_cot_text, max_new_tokens=max_new_tokens_budget)
        
        if not layer1_candidates:
            return "", client_states

        # We now have N potential futures.
        # We need to evaluate V(S_i) for each.
        
        state_values = [] # (index, score)
        state_details = [] # stats for logging

        # Optimization: If a candidate contains \boxed{}, it's a terminal state.
        # We should probably prioritize it or treat it specially.
        # For now, let's just evaluate it normally (generation might fail or return nothing, 
        # but scoring handles that).
        

        for i, cand_r in enumerate(layer1_candidates):
            # Hypothetical State S_i
            hypothetical_history = history_parts + [cand_r]
            hypothetical_text = "\n\n".join(hypothetical_history)
            
            # --- Dynamic Token Budget Calculation (Layer 2) ---
            # Re-check budget for the hypothetical state
            curr_len_2_tokens = current_len_tokens # Base approximation
            
            # Use first client to estimate token length of the new candidate part
            client_for_tokenization = engine.clients[0] if engine.clients else None
            
            if client_for_tokenization:
                 try:
                     cand_len = client_for_tokenization.get_token_len(cand_r)
                     curr_len_2_tokens = current_len_tokens + cand_len
                 except Exception:
                     curr_len_2_tokens = current_len_tokens + (len(cand_r) // 3)
            else:
                 curr_len_2_tokens = current_len_tokens + (len(cand_r) // 3)

            rem_tokens_2 = soft_limit_tokens - curr_len_2_tokens
            if rem_tokens_2 <= 0:
                budget_2 = 1
            else:
                 budget_2 = min(1024, int(rem_tokens_2))

            # --- Step 2: Lookahead (Second Layer Generation) ---
            # Generate M responses based on S_i
            # Since we don't have KV cache efficient cloning yet, this will be slow (re-encoding S_i).
            layer2_candidates = engine.generate_candidates(problem, hypothetical_text, max_new_tokens=budget_2)
            
            if not layer2_candidates:
                # If no one can continue, this might be a bad state? 
                # Or a finished state?
                if "\\boxed{" in cand_r:
                    # It's a terminal state, give it a high bonus if it's consistent?
                    # For simplicity, if terminal, we assign a heuristic score (0.0 implies perfect?)
                    # Let's treating it as "Perfect Consensus" if it terminates.
                    logging.info(f"    State {i} is terminal.")
                    state_values.append(0.0) 
                    continue
                else:
                    # Dead end
                    state_values.append(float('inf'))
                    continue

            # --- Step 3: Evaluation (Second Layer Scoring with Info Gain) ---
            # Use the new score_candidates_with_gain method
            layer2_metrics = engine.score_candidates_with_gain(
                base_messages,
                hypothetical_text,
                layer2_candidates
            )
            
            # Aggregate scores for S_i
            # V(S_i) = Aggregation of scores of layer2 steps.
            
            layer2_step_scores = []
            s_full_log = []
            s_blind_log = []

            for m_idx, metrics in enumerate(layer2_metrics):
                full_nll = metrics['full']
                blind_nll = metrics['blind']
                lookahead_cand = layer2_candidates[m_idx]
                
                s_full_log.append(full_nll)
                s_blind_log.append(blind_nll)

                # Formula: Score = NLL_Full - lambda * (NLL_Blind - NLL_Full)
                # Lower score is better.
                # If NLL_Blind is high (good info gain), score decreases (improves).
                
                # Apply Info Weight
                # To prevent instability if blind NLL implies confusion (blind < full), 
                # we can clamp gain to >= 0 or just trust the raw value.
                # Raw value: (1+lambda)*full - lambda*blind
                base_score = (1.0 + self.info_weight) * full_nll - self.info_weight * blind_nll

                # Alpha-NLL / Length Penalty Logic
                # value_metric now primarily controls how we handle LENGTH.
                # If metric is "mean_nll", we just use base_score (which is mean).
                
                est_tokens = max(1, len(lookahead_cand) / 4.0)
                
                if self.value_metric == "sum_nll":
                    final_score = base_score * est_tokens
                elif self.value_metric == "alpha_nll":
                    # Alpha-NLL: Score * Length^(1-alpha)
                    penalty_factor = est_tokens ** (1.0 - self.alpha)
                    final_score = base_score * penalty_factor
                else:
                    # Default mean_nll
                    final_score = base_score

                layer2_step_scores.append(final_score)
            
            if not layer2_step_scores:
                state_values.append(float('inf'))
                state_details.append({"full": -1, "blind": -1})
            else:
                # The value of State S_i is the "easiness" of the best path forward, 
                # OR the average "easiness" of all paths?
                # "Easiness" = Low NLL.
                state_values.append(np.mean(layer2_step_scores))
                state_details.append({
                    "full": np.mean(s_full_log) if s_full_log else -1,
                    "blind": np.mean(s_blind_log) if s_blind_log else -1
                })


        # --- Step 4: Selection ---
        if not state_values or min(state_values) == float('inf'):
             logging.info(" No valid futures. Fallback to greedy/first.")
             best_idx = 0
             # Fallback check
             if not layer1_candidates:
                 return "", client_states
        else:
             best_idx = np.argmin(state_values)
        
        best_step = layer1_candidates[best_idx]
        best_val = state_values[best_idx] if best_idx < len(state_values) else -1
        
        logging.info("Candidates (Values):")
        for i, cand in enumerate(layer1_candidates):
            preview = cand.replace('\\n', ' ')
            marker = "*" if i == best_idx else " "
            val = state_values[i] if i < len(state_values) else -1
            
            # Detail log
            det = state_details[i] if i < len(state_details) else {}
            # Format: [V=1.23, F=1.5, B=2.0]
            # V is the final score (lower is better), F is Full NLL, B is Blind NLL
            detail_str = f"[V={val:.3f}, F={det.get('full',-1):.3f}, B={det.get('blind',-1):.3f}]"
            
            logging.info(f"  {detail_str} {marker} {preview}")

        # 5. Update Real States
        new_history = history_parts + [best_step]
        new_history_text = "\n\n".join(new_history)
        new_client_states = engine.update_states(base_messages, new_history_text)
        
        return best_step, new_client_states
