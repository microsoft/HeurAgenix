import re
import os
from typing import Dict, List
from datasets import load_dataset, load_from_disk
from src.tasks.base_task import BaseTask
from src.util.math_grading import grade_answer

class AimeTask(BaseTask):
    def __init__(self, subset: str = "2024", system_prompt: str = None):
        """
        Task Adapter for AIME (2024 & 2025).
        Args:
            subset: "2024" (HuggingFaceH4) or "2025" (opencompass)
        """
        self.subset = str(subset)
        self.data = None
        
        # Determine Dataset ID based on subset
        if self.subset == "2025":
            self.dataset_id = "opencompass/AIME2025"
        else:
            self.dataset_id = "HuggingFaceH4/aime_2024" # Default to 2024

        # System Prompt (Consolidated for AIME format)
        if system_prompt is None:
            self.system_prompt = (
                "You are a helpful assistant who is good at mathematics. "
                "Please solve the problem step by step. "
                "CRITICAL: You must enclose every individual logical step within <step> and </step> tags. "
                "Do not output any text outside of these tags. "
                "CRITICAL: Each step must contain substantive mathematical reasoning or calculation. "
                "Do NOT split a single logical step into multiple tags. "
                "At the end of your solution, you MUST put the final answer inside \\boxed{}. "
                "The answer should typically be a non-negative integer between 000 and 999."
                "For example: "
                "<step>First, we calculate ...</step>"
                "<step>The answer is \\boxed{365}.</step>"
            )
        else:
            self.system_prompt = system_prompt

    def get_dataset(self) -> List[Dict]:
        if self.data is None:
            # Check for AMLT environment to load dataset from disk
            amlt_data_dir = os.getenv("AMLT_DATA_DIR")
            dataset = None

            if amlt_data_dir:
                 dataset_path = os.path.join(amlt_data_dir, self.dataset_id)
                 # AIME datasets usually use 'train' split
                 try:
                     loaded = load_from_disk(dataset_path)
                     if hasattr(loaded, 'keys') and 'train' in loaded:
                         dataset = loaded['train']
                     else:
                         dataset = loaded
                 except Exception:
                     # Fallback to online if disk load fails
                     dataset = None
            
            if dataset is None:
                if self.subset == "2025":
                    # Special handling for opencompass/AIME2025 (Config based, 'test' split)
                    from datasets import concatenate_datasets
                    ds1 = load_dataset(self.dataset_id, "AIME2025-I", split='test')
                    ds2 = load_dataset(self.dataset_id, "AIME2025-II", split='test')
                    dataset = concatenate_datasets([ds1, ds2])
                else:
                    dataset = load_dataset(self.dataset_id, split='train')

            self.data = []
            
            for item in dataset:
                # NORMALIZE COLUMNS
                # 2024: 'problem', 'answer', 'solution'
                # 2025: 'question', 'answer'
                
                problem_text = item.get("problem") or item.get("question")
                if not problem_text:
                    continue # Valid skip
                    
                ground_truth = str(item.get("answer", "")).strip()
                
                self.data.append({
                    "problem": problem_text,
                    "ground_truth": ground_truth,
                    "id": item.get("id", "unknown"),
                    "full_solution": item.get("solution", "") # 2025 may not have full solution
                })
        return self.data

    def extract_answer(self, response: str) -> str:
        """
        Standard Boxed Extraction
        """
        return self._extract_boxed_content(response)

    def _extract_boxed_content(self, text: str) -> str:
        if "\\boxed{" not in text:
            # Fallback: finding "Answer: X" for weak models
            match = re.search(r'(?:Answer|The answer is)\s*[:\s]\s*([0-9]+)', text, re.IGNORECASE)
            if match:
                return match.group(1)
            return ""
            
        start_indices = [m.start() for m in re.finditer(r'\\boxed\{', text)]
        if not start_indices:
            return ""
        
        for start_idx in reversed(start_indices):
            content_start = start_idx + 7
            balance = 1
            for i in range(content_start, len(text)):
                char = text[i]
                if char == '{':
                    balance += 1
                elif char == '}':
                    balance -= 1
                
                if balance == 0:
                    return text[content_start:i]
        return ""

    def verify_answer(self, prediction: str, ground_truth: str) -> bool:
        return grade_answer(prediction, ground_truth)
