import numpy as np
from typing import List
from src.util.llm_client.local_model_client import LocalModelClient
from src.util.text_utils import smart_split_steps

class ConsensusEngine:
    def __init__(self, client_config_paths: List[str], system_prompt: str = None):
        self.clients: List[LocalModelClient] = []
        self.system_prompt = system_prompt
        for config_path in client_config_paths:
            client = LocalModelClient(config_path, system_prompt=system_prompt)
            self.clients.append(client)

    def decide(self, problem: str, max_step: int=20) -> str:
        """
        Iterative Step-level Consensus Decision Making.
        Generates the solution step-by-step, validating each step across all models.
        """
        
        # 1. Base Context (User Message)
        # We hold the static context (System + User) separate from the generated CoT.
        if self.system_prompt:
            base_messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": problem}]
        else:
            base_messages = [{"role": "user", "content": problem}]
        
        # 2. State Tracking
        final_response_parts = []
        # current_cot_text = "" (Removed, using final_response_parts)
        
        # Cache for each client to perform subtraction (Get NLL of just the new step)
        # Store: {'nll_sum': float, 'token_len': int}
        client_states = [{'nll_sum': 0.0, 'token_len': 0} for _ in self.clients]
        # Debug: print problem number
        print(f"\n[ConsensusEngine] Starting processing for problem: {problem}", flush=True)

        for step_idx in range(max_step):
            # Debug: print step number
            print(f"\n--- Step {step_idx + 1} ---", flush=True)
            
            # --- Phase 1: Diversified Generation ---
            step_candidates = []
            
            # Reconstruct current CoT text from parts
            current_cot_text = "\n\n".join(final_response_parts)

            for i, client in enumerate(self.clients):
                # Prepare context for the client
                client.reset(self.system_prompt)
                client.add_message(problem, role="user")
                
                # IMPORTANT: For generation, we DO NOT add history as a message.
                # Instead, we pass it as a `continue_prefix`.
                # This forces the model to treat it as "partially generated text" and continue flow.
                
                try:
                    # Generate completion
                    # We pass current_cot_text as prefix
                    prefix = current_cot_text + "\n\n" if current_cot_text else None
                    
                    full_response = client.chat(continue_prefix=prefix)
                    
                    if not full_response:
                        continue
                        
                    # Extract the NEW part. 
                    new_steps = smart_split_steps(full_response)
                    if new_steps:
                        # We only take the first step as the candidate
                        candidate_step = new_steps[0]
                        step_candidates.append(candidate_step)
                    else:
                        step_candidates.append(full_response.strip())
                         
                except Exception as e:
                    print(f"Agent {i} failed to generate: {e}", flush=True)

            if not step_candidates:
                print("No candidates generated in this step. Aborting.", flush=True)
                break

            
            # --- Phase 2: Cross-Consistency Evaluation (Step NLL) ---
            candidate_scores = [] # store avg NLL (smaller is better)
            
            for cand in step_candidates:
                # Construct temporary full text for evaluation
                # Join history + candidate
                candidate_parts = final_response_parts + [cand]
                candidate_full_text = "\n\n".join(candidate_parts)
                
                total_step_nll = 0
                valid_reviewers = 0
                
                for j, reviewer in enumerate(self.clients):
                    try:
                        # 1. Get NLL of the WHOLE sequence (accumulated + new)
                        avg_nll = reviewer.get_sequence_score(base_messages, candidate_full_text)
                        
                        # 2. Get Length of the WHOLE sequence response
                        # add_special_tokens=False is important
                        tokens = reviewer.pipeline.tokenizer(candidate_full_text, add_special_tokens=False, return_tensors="pt").input_ids
                        full_len = tokens.shape[1]
                        
                        full_nll_sum = avg_nll * full_len
                        
                        # 3. Subtract Previous Info to isolate Step NLL
                        prev_nll_sum = client_states[j]['nll_sum']
                        prev_len = client_states[j]['token_len']
                        
                        step_len = full_len - prev_len
                        
                        if step_len > 0:
                            # Step NLL = (Total NLL Sum - Prev NLL Sum) / Step Len
                            step_nll = (full_nll_sum - prev_nll_sum) / step_len
                            # Clip negative NLL (precision errors)
                            step_nll = max(0.0, step_nll)
                            
                            total_step_nll += step_nll
                            valid_reviewers += 1
                        else:
                            pass
                            
                    except Exception as e:
                        print(f"Reviewer {j} eval failed: {e}", flush=True)
                
                final_score = total_step_nll / valid_reviewers if valid_reviewers > 0 else float('inf')
                candidate_scores.append(final_score)

            # --- Phase 3: Selection ---
            if not candidate_scores or min(candidate_scores) == float('inf'):
                # Debug: print information.
                print("No more candidates, choose first one:", flush=True)
                for cand in step_candidates:
                    preview = cand.replace('\\n', ' ')
                    print(f"  {preview}", flush=True)
                best_step = step_candidates[0]
            else:
                best_idx = np.argmin(candidate_scores)
                best_step = step_candidates[best_idx]
                best_score = candidate_scores[best_idx]
                # Debug: print information.
                print("Candidates:", flush=True)
                for index, cand in enumerate(step_candidates):
                    preview = cand.replace('\\n', ' ')
                    if index == best_idx:
                        print(f"  [{best_score:.4f}, *] {preview}", flush=True)
                    else:
                        print(f"  [{best_score:.4f}] {preview}", flush=True)

            # --- Phase 4: Update State ---
            final_response_parts.append(best_step)
            current_cot_text = "\n\n".join(final_response_parts)
            
            # Update client_states for the *chosen* path
            for j, reviewer in enumerate(self.clients):
                try:
                    avg_nll = reviewer.get_sequence_score(base_messages, current_cot_text)
                    tokens = reviewer.pipeline.tokenizer(current_cot_text, add_special_tokens=False, return_tensors="pt").input_ids
                    full_len = tokens.shape[1]
                    
                    client_states[j]['nll_sum'] = avg_nll * full_len
                    client_states[j]['token_len'] = full_len
                except:
                    pass

            # --- Phase 5: Termination Check ---
            if "\\boxed{" in best_step:
                # Debug: print finished.
                print("Termination condition (boxed) met.", flush=True)
                break
        
        return current_cot_text
