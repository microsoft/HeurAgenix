from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from src.engine.engine import SwarmEngine

class BaseStrategy(ABC):
    """
    Layer 3: Strategy Interface
    Abstract base class for consensus strategies.
    """
    
    @abstractmethod
    def select_next_step(
        self, 
        problem: str, 
        history_parts: List[str], 
        engine: SwarmEngine,
        client_states: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """
        Decides the next reasoning step.
        
        Args:
            problem: The original problem text.
            history_parts: List of previous accepted steps.
            engine: The SwarmEngine instance for execution.
            client_states: Current NLL/Len states for delta calculation.
            
        Returns:
            Tuple[selected_step_text, new_client_states]
        """
        pass
