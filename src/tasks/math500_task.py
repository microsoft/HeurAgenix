import re
from typing import Dict, List, Optional
from datasets import load_dataset
from src.tasks.base_task import BaseTask

class Math500Task(BaseTask):
    def __init__(self, subset: str = "test"):
        """
        Args:
            subset: The split to load (default: "test" for MATH-500)
        """
        self.dataset_name = "HuggingFaceH4/MATH-500"
        self.subset = subset
        self.data = None

    def get_dataset(self) -> List[Dict]:
        if self.data is None:
            # MATH-500 usually has a 'test' split
            ds = load_dataset(self.dataset_name, split=self.subset)
            self.data = []
            for item in ds:
                # MATH-500 structure: 'problem', 'solution', 'answer', 'subject', 'level'
                self.data.append({
                    "problem": item["problem"],
                    "ground_truth": item["answer"],  # The short answer inside boxed
                    "full_solution": item["solution"], # The full step-by-step solution
                    "subject": item.get("subject", "math"),
                    "level": item.get("level", "unknown")
                })
        return self.data

    def format_prompt(self, problem_data: Dict) -> List[Dict]:
        """
        Modified to include a system prompt enforcing the output format.
        Most modern math models (DeepSeek, Qwen, Llama3) perform better with a system prompt.
        """
        system_content = (
            "You are a helpful assistant who is good at mathematics. "
            "Please solve the problem step by step. "
            "At the end of your solution, you MUST put the final answer inside \\boxed{}. "
            "For example: The answer is \\boxed{5}."
        )
        user_content = f"Problem:\n{problem_data['problem']}"
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    def extract_answer(self, response: str) -> str:
        """
        Extract the last \boxed{...} content.
        This handles nested braces to some extent by using a greedy match or external libraries if needed,
        but a simple regex is often 'good enough' for standard outputs if the model is compliant.
        """
        # Finds all \boxed{...} patterns. 
        # Note: This simple regex fails on nested braces like \boxed{\frac{1}{2}}.
        # A more robust extractor is usually needed for complex LaTeX.
        # But for now, let's use a slightly improved regex or fallback to simple search.
        
        # Strategy 1: Simple Regex (non-nested)
        # matches = re.findall(r'\\boxed\{(.*?)\}', response)
        
        # Strategy 2: Bracket counting (robust for nesting)
        return self._extract_boxed_content(response)

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
            
        last_boxed_response = ""
        
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
        Symbolic verification is hard. We use string normalization equality here.
        Ideally, use sympy or specialized math equivalence checkers.
        """
        norm_pred = self._normalize_answer(prediction)
        norm_gt = self._normalize_answer(ground_truth)
        return norm_pred == norm_gt

    def _normalize_answer(self, s: str) -> str:
        if not s:
            return ""
        s = str(s).strip()
        # Remove common LaTeX wrappers that don't change value for simple comparison
        # e.g., \text{4} -> 4, \mathrm{cm} -> cm
        s = re.sub(r'\\text\{([^}]+)\}', r'\1', s)
        s = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', s)
        
        # Remove whitespace
        s = s.replace(" ", "")
        
        # Simple fractions normalization: \frac{1}{2} -> 1/2 (optional, depends on model output)
        
        return s
