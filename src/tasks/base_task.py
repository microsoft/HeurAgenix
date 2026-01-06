from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseTask(ABC):
    """
    Abstract base class for all tasks (datasets).
    Responsible for loading data, formatting prompts, and evaluating results.
    """
    
    @abstractmethod
    def get_dataset(self) -> List[Dict]:
        """
        Load and return the dataset as a list of dictionaries.
        Each item should at least contain the raw 'problem' and 'ground_truth'.
        """
        pass

    @abstractmethod
    def format_prompt(self, problem_data: Dict) -> List[Dict]:
        """
        Convert a problem data item into a chat message list (system, user).
        Returns:
            List[Dict]: e.g. [{"role": "user", "content": "..."}]
        """
        pass

    @abstractmethod
    def extract_answer(self, response: str) -> str:
        """
        Extract the core answer (e.g., content inside \boxed{}) from the model's full response.
        """
        pass
    
    @abstractmethod
    def verify_answer(self, prediction: str, ground_truth: str) -> bool:
        """
        Compare the extracted prediction with the ground truth.
        """
        pass
