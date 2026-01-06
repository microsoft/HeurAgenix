from typing import List, Dict, Any
from src.util.llm_client.base_llm_client import BaseLLMClient
from src.util.llm_client.local_model_client import LocalModelClient

class ConsensusEngine:
    def __init__(self, client_configs: List[Dict], output_dir: str = None):
        self.clients: List[BaseLLMClient] = []
        for config in client_configs:
            # Initialize clients based on config
            # Currently defaults to LocalModelClient
            # TODO: Add logic to choose different clients based on config['type']
            client = LocalModelClient(config, output_dir=output_dir)
            self.clients.append(client)
        
        print(f"Initialized ConsensusEngine with {len(self.clients)} clients.")

    def decide(self, messages: List[Dict]) -> str:
        """
        Main entry point for the consensus mechanism.
        
        Args:
            messages: The context/prompt messages.
            
        Returns:
            The best response text selected by the engine.
        """
        # Placeholder logic: just use the first client for now
        if not self.clients:
            return "No clients available."
        
        # --- PLACEHOLDER START ---
        # In the real implementation, this will:
        # 1. Parallel Generate from all self.clients (Phase I)
        # 2. Peer Review / NLL Scoring (Phase II)
        # 3. Aggregate scores and select best (Phase III)
        
        # For now, to test the pipeline, we just use the first client.
        primary_client = self.clients[0]
        
        # Important: We must not modify the state of the client permanently if we can avoid it,
        # but for LocalModelClient as written, we set .messages.
        primary_client.messages = messages 
        
        # print(f"DEBUG: ConsensusEngine delegating to {primary_client.name} (Placeholder)")
        return primary_client.chat_once()
        # --- PLACEHOLDER END ---
