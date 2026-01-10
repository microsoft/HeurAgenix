import torch
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Any
from src.engine.llm_client.local_model_client import LocalModelClient
from src.util.text_utils import smart_split_steps

class SwarmEngine:
    """
    Layer 2: Compute Layer
    Manages the swarm of LLM clients and provides parallelized atomic operations.
    """
    def __init__(self, client_config_paths: List[str], system_prompt: str = None):
        self.clients: List[LocalModelClient] = []
        self.system_prompt = system_prompt
        
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        print(f"Detected {num_gpus} GPUs. Assigning clients round-robin.")

        for i, config_path in enumerate(client_config_paths):
            device_id = i % num_gpus
            print(f"Initializing Client {i} on device {device_id}")
            client = LocalModelClient(config_path, system_prompt=system_prompt, device_id=device_id)
            self.clients.append(client)

    def generate_candidates(self, problem: str, current_cot_text: str) -> List[str]:
        """
        Phase 1: Diversified Generation
        Broadcasts the generation request to all clients.
        Returns a list of candidate steps (one per client).
        """
        step_candidates = []

        def _generate(client_idx, client_inst):
            client_inst.reset(self.system_prompt)
            client_inst.add_message(problem, role="user")
            
            try:
                # We pass current_cot_text as prefix
                prefix = current_cot_text + "\n\n" if current_cot_text else None
                full_response = client_inst.chat(continue_prefix=prefix)
                
                if not full_response:
                    return None
                    
                # Extract the NEW part using smart split
                new_steps = smart_split_steps(full_response)
                if new_steps:
                    return new_steps[0]
                else:
                    return full_response.strip()
            except Exception as e:
                print(f"Agent {client_idx} failed to generate: {e}", flush=True)
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            futures = [executor.submit(_generate, i, c) for i, c in enumerate(self.clients)]
            for f in concurrent.futures.as_completed(futures):
                res = f.result()
                if res:
                    step_candidates.append(res)
        
        return step_candidates

    def score_candidates(
        self, 
        base_messages: List[Dict], 
        history_parts: List[str], 
        candidates: List[str], 
        client_states: List[Dict]
    ) -> List[List[Optional[float]]]:
        """
        Phase 2: Cross-Consistency Evaluation (Step NLL)
        Scores each candidate against all clients (Peer Review).
        
        Args:
            base_messages: The prompt context (System + User).
            history_parts: Existing CoT steps.
            candidates: List of new candidate steps to evaluate.
            client_states: List of dicts {'nll_sum': ..., 'token_len': ...} for each client.
            
        Returns:
            A matrix of scores: result[cand_idx][reviewer_idx] = Step NLL (or None if failed).
        """
        
        # We process (Candidate, Reviewer) pairs.
        # To maximize throughput, we can flatten tasks or loop.
        # Since we have N clients, usually N is small (2-4).
        # We can iterate candidates and parallelize reviewers.

        scores_matrix = [[None for _ in range(len(self.clients))] for _ in range(len(candidates))]

        # Pre-compute full texts for each candidate
        candidate_full_texts = []
        for cand in candidates:
            parts = history_parts + [cand]
            candidate_full_texts.append("\n\n".join(parts))

        # We define a single scoring task
        def _score_task(reviewer_idx, reviewer_inst, cand_idx, full_text):
            try:
                # 1. Get NLL of WHOLE sequence
                avg_nll = reviewer_inst.get_sequence_score(base_messages, full_text)
                
                # 2. Get Length
                tokens = reviewer_inst.pipeline.tokenizer(full_text, add_special_tokens=False, return_tensors="pt").input_ids
                full_len = tokens.shape[1]
                full_nll_sum = avg_nll * full_len
                
                # 3. Delta (Step NLL)
                prev_nll_sum = client_states[reviewer_idx]['nll_sum']
                prev_len = client_states[reviewer_idx]['token_len']
                
                step_len = full_len - prev_len
                
                if step_len > 0:
                    step_nll = (full_nll_sum - prev_nll_sum) / step_len
                    return max(0.0, step_nll)
                return None
            except Exception as e:
                # print(f"Reviewer {reviewer_idx} eval failed: {e}", flush=True)
                return None

        # Execute
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            futures = {}
            for c_idx, text in enumerate(candidate_full_texts):
                for r_idx, client in enumerate(self.clients):
                    # Exclude self-score if needed? (Logic parameter, better handled in Strategy)
                    # For Engine, we just compute all. Strategy filters.
                    f = executor.submit(_score_task, r_idx, client, c_idx, text)
                    futures[f] = (c_idx, r_idx)
            
            for f in concurrent.futures.as_completed(futures):
                c_idx, r_idx = futures[f]
                res = f.result()
                scores_matrix[c_idx][r_idx] = res

        return scores_matrix

    def update_states(self, base_messages: List[Dict], history_text: str) -> List[Dict]:
        """
        Phase 4: State Update
        re-calculates the baselines (NLL sum, len) for the chosen history.
        """
        new_states = [{'nll_sum': 0.0, 'token_len': 0} for _ in range(len(self.clients))]
        
        def _update(idx, client):
            try:
                avg_nll = client.get_sequence_score(base_messages, history_text)
                tokens = client.pipeline.tokenizer(history_text, add_special_tokens=False, return_tensors="pt").input_ids
                full_len = tokens.shape[1]
                return (avg_nll * full_len, full_len)
            except:
                return (0.0, 0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            futures = {executor.submit(_update, i, c): i for i, c in enumerate(self.clients)}
            for f in concurrent.futures.as_completed(futures):
                i = futures[f]
                nll_sum, t_len = f.result()
                new_states[i]['nll_sum'] = nll_sum
                new_states[i]['token_len'] = t_len
                
        return new_states
