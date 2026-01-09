import re
import os
from typing import Dict, List
from datasets import load_dataset, load_from_disk
from src.tasks.base_task import BaseTask
from src.util.math_grading import grade_answer

class Math500Task(BaseTask):
    def __init__(self, subset: str = "test", system_prompt: str = None):
        """
        Args:
            subset: The split to load (default: "test" for MATH-500)
        """
        self.dataset_name = "HuggingFaceH4/MATH-500"
        self.subset = subset
        self.data = None
        if system_prompt is None:
            self.system_prompt = (
                "You are a helpful assistant who is good at mathematics. "
                "Please solve the problem step by step. "
                "CRITICAL: You must enclose every individual logical step within <step> and </step> tags. "
                "Do not output any text outside of these tags. "
                "At the end of your solution, you MUST put the final answer inside \\boxed{}. "
                "For example: "
                "<step>First, we calculate...</step>"
                "<step>The answer is \\boxed{5}.</step>"
            )
        else:
            self.system_prompt = system_prompt

    def get_dataset(self) -> List[Dict]:
        if self.data is None:
            # Check for AMLT environment to load dataset from disk
            amlt_data_dir = os.getenv("AMLT_DATA_DIR")
            if amlt_data_dir:
                 dataset_path = os.path.join(amlt_data_dir, self.dataset_name)
                 dataset = load_from_disk(dataset_path)[self.subset]
            else:
                 dataset = load_dataset(self.dataset_name, split=self.subset)

            self.data = []
            for item in dataset:
                # MATH-500 structure: 'problem', 'solution', 'answer', 'subject', 'level'
                self.data.append({
                    "problem": item["problem"],
                    "ground_truth": item["answer"],  # The short answer inside boxed
                    "full_solution": item["solution"], # The full step-by-step solution
                    "subject": item.get("subject", "math"),
                    "level": item.get("level", "unknown")
                })
        return self.data


    def extract_answer(self, response: str) -> str:
        """
        Extract the last \boxed{...} content.
        Also attempts to extract answer following "The answer is" pattern if \boxed{} is missing.
        """
        # 1. Try extracting \boxed{...} (Priority)
        boxed_content = self._extract_boxed_content(response)
        if boxed_content:
            return boxed_content

        # 2. Fallback: Look for "The answer is: <content>" or similar patterns
        # DeepSeek often outputs: "The answer is: $(3,\frac{\pi}{2})$"
        # We look for the last occurrence of "answer is" and take the rest of the line or sentence
        patterns = [
            r"answer is[:\s]+(.*?)(?:\n|$|\.)",
            r"answer is[:\s]+\$(.*?)\$",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                # Take the last match as it's usually the conclusion
                candidate = matches[-1].strip()
                # Remove trailing period if present
                if candidate.endswith('.'):
                    candidate = candidate[:-1]
                return candidate
                
        return ""

    def _extract_boxed_content(self, text: str) -> str:
        """
        Helper to extract content inside the last \boxed{...}, handling nested braces.
        """
        if "\\boxed{" not in text:
            return ""
            
        # Find all indices of \boxed{
        start_indices = [m.start() for m in re.finditer(r'\\boxed\{', text)]
        
        # We generally want the *last* boxed answer in the text
        if not start_indices:
            return ""
        
        # Iterate backwards to find the last valid boxed content
        for start_idx in reversed(start_indices):
            content_start = start_idx + 7 # len("\boxed{")
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
        """
        Uses robust grading logic to verify answer correctness.
        Delegates to src.util.math_grading.grade_answer
        """
        return grade_answer(prediction, ground_truth)
