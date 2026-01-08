from typing import List, Dict, Any, Tuple
from src.util.llm_client.base_llm_client import BaseLLMClient
from src.util.llm_client.local_model_client import LocalModelClient

class ConsensusEngine:
    def __init__(self, client_config_paths: List[str], output_dir: str = None):
        self.clients: List[BaseLLMClient] = []
        for config_path in client_config_paths:
            # Initialize clients based on config
            # Currently defaults to LocalModelClient
            # TODO: Add logic to choose different clients based on config['type']
            client = LocalModelClient(config_path, output_dir=output_dir)
            self.clients.append(client)
        
        print(f"Initialized ConsensusEngine with {len(self.clients)} clients.")

    def decide(self, messages: List[Dict]) -> Tuple[str, int]:
        """
        Main entry point for the consensus mechanism.
        
        Args:
            messages: The context/prompt messages.
            
        Returns:
            The best response text selected by the engine with target client index.
        """
        import copy

        if not self.clients:
            return "No clients available.", -1
        
        # 1. Generate responses from all agents
        responses = []

        print(f"Starting consensus generation with {len(self.clients)} agents...")

        for i, client in enumerate(self.clients):
            # Use set_history to safely copy and set the context for the agent
            client.set_history(messages)
            try:
                # chat() returns the response content and appends to client.messages
                # We use chat() to get retry logic handling
                response_content = client.chat()
                if response_content:
                    responses.append(response_content)
                else:
                    print(f"Agent {i} ({client.name}) returned no content.")
                    responses.append("")
            except Exception as e:
                print(f"Agent {i} ({client.name}) failed to generate: {e}")
                responses.append("")

        # filter out empty responses for checking if we have any valid response
        valid_responses = [r for r in responses if r]
        if not valid_responses:
            return "All agents failed to generate a response."
            
        # 2. Cross-Consistency Evaluation
        # Score(R_i) = Average NLL evaluated by peer agents
        
        scores = [0] * len(responses)
        for i, response_i in enumerate(responses):
            nll_sum = 0
            count = 0
            
            for j, client in enumerate(self.clients):
                # Skip self-evaluation (Cross-Consistency)
                if i == j:
                    continue

                try:
                    # We pass the ORIGINAL PROMPT (messages) and the CANDIDATE RESPONSE (response_i)
                    # get_sequence_score calculates NLL(response | context)
                    nll = client.get_sequence_score(messages, response_i)
                    nll_sum += nll
                    count += 1
                except Exception as e:
                    print(f"Agent {j} ({client.name}) failed to evaluate Response {i}: {e}")
            
            if count > 0:
                avg_nll = nll_sum / count
                scores[i] = -avg_nll  # Lower NLL is better

        # 3. Consensus Aggregation
        # Find the index with the maximum score (Lowest NLL = Best consistency)
        best_idx = -1
        max_score = float('-inf')
        
        for i, score in enumerate(scores):
            # print(f"Response {i}: Score (NLL) = {score:.4f}")
            if score > max_score:
                max_score = score
                best_idx = i
        
        if best_idx != -1:
            best_response = responses[best_idx]
            print(f"Selected Response {best_idx} with maximal NLL score {max_score:.4f}")
            return best_response, best_idx
        else:
            return valid_responses[0], 0
