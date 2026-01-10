
import re
from typing import List

def smart_split_steps(text: str) -> List[str]:
    """
    Splits Chain-of-Thought text into steps based on newlines, 
    but ensures that LaTeX math blocks and environments are not split.
    
    Rules:
    1. FIRST PRIORITY: If <step>...</step> tags are present, use them.
    2. Do not split inside $$ ... $$
    3. Do not split inside \begin{...} ... \end{...}
    4. Do not split inside $ ... $ (though usually these are inline and don't contain newlines, but just in case)
    5. Split on single or multiple \n outside of these blocks.
    """
    
    # 0. Check for explicit <step> tags
    # We use re.DOTALL to let . match newlines inside the tag
    # CHANGED: We now capture the full tag <step>...</step> so that the context 
    # passed back to the model (continue_prefix) preserves the tags.
    step_tags = re.findall(r'(<step>.*?</step>)', text, re.DOTALL)
    if len(step_tags) > 0:
        # Found structured steps, verify they are not just empty
        valid_steps = [s.strip() for s in step_tags if s.strip()]
        if valid_steps:
           return valid_steps
           
    # Fallback to heuristic splitting if no tags found
    steps = []
    current_step = []
    
    # State flags
    in_dollar = False # $ ... $
    in_double_dollar = False # $$ ... $$
    env_stack = [] # Stack for \begin{...} ... \end{...} environments
    
    i = 0
    length = len(text)
    
    while i < length:
        # Check for double dollar $$ ... $$
        if i + 1 < length and text[i:i+2] == '$$':
            # Toggle state
            in_double_dollar = not in_double_dollar
            current_step.append('$$')
            i += 2
            continue
            
        # Check for single dollar $ ... $ 
        # (Only if not inside double dollar)
        if text[i] == '$' and not in_double_dollar:
            # Handle escaped dollar \$
            if i > 0 and text[i-1] == '\\':
                current_step.append('$')
            else:
                in_dollar = not in_dollar
                current_step.append('$')
            i += 1
            continue

        # Check for \begin{...}
        # Only check if strictly not in dollar modes (although latex allows nested, 
        # usually we care about top level splits)
        # But CoT often puts \begin inside $$.
        # If we are in $$, we definitely don't split on \n.
        # So we only need to track \begin environments if they appear OUTSIDE of $$ 
        # (which is valid LaTeX, e.g. \begin{equation})
        
        match_begin = None
        if not (in_dollar or in_double_dollar):
            match_begin = re.match(r'^\\begin\{([^}]+)\}', text[i:])
        
        if match_begin:
            env_name = match_begin.group(1)
            env_stack.append(env_name)
            tag = match_begin.group(0)
            current_step.extend(list(tag))
            i += len(tag)
            continue
            
        # Check for \end{...}
        match_end = None
        if not (in_dollar or in_double_dollar):
            match_end = re.match(r'^\\end\{([^}]+)\}', text[i:])
            
        if match_end:
            env_name = match_end.group(1)
            # Pop if matches (or just pop blindly to be robust against malformed text)
            if env_stack:
                if env_stack[-1] == env_name:
                    env_stack.pop()
                else:
                    # Mismatch or nested weirdness, just pop? or ignore?
                    # Let's simple pop if something exists
                    pass
            
            tag = match_end.group(0)
            current_step.extend(list(tag))
            i += len(tag)
            continue

        # Check for valid split point (\n)
        if text[i] == '\n':
            # SAFE TO SPLIT CONDITION:
            # Not in $ or $$
            # Not inside an environment stack
            is_safe = (not in_dollar) and (not in_double_dollar) and (len(env_stack) == 0)
            
            if is_safe:
                # Flush current step
                content = "".join(current_step).strip()
                if content:
                    steps.append(content)
                current_step = []
                
                # Skip subsequent newlines (consume them)
                while i < length and text[i] == '\n':
                    i += 1
                continue
        
        # Default behavior: add char
        current_step.append(text[i])
        i += 1
        
    # Flush remaining
    content = "".join(current_step).strip()
    if content:
        steps.append(content)
        
    return steps
