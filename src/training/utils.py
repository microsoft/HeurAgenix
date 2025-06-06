import re

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# List of known algorithm names, potentially with specific identifiers.
ALGORITHM_NAMES = [
    "nearest_insertion_c1f0",
    "random_80a0",
    "farthest_insertion_b6d3",
    "greedy_randomized_adaptive_search_procedure_grasp_5a6a",
    "ant_colony_d4f7",
    "cheapest_insertion_605f",
    "simulated_annealing_e625",
    "random_successive_insertion_57b4",
    "_2opt_89aa",
    "insertion_heuristics_050b",
    "_3opt_e75b",
    "greedy_algorithm_3ca7",
    "k_nearest_neighbors_insertion_9e8b",
    "random_pairwise_insertion_7493", # Added missing comma here
    "three_opt_e8d7",
    "two_opt_0554",
    "node_shift_between_routes_7b8a",
    "farthest_insertion_4e1d",
    "greedy_f4c4",
    "min_cost_insertion_7bfa",
    "nearest_neighbor_99ba",
    "petal_algorithm_b384",
    "random_bfdc",
    "saving_algorithm_710e",
    "variable_neighborhood_search_614b",
]

def get_state_card(text: str) -> str | None:
    """
    Extracts the content following "[state card]:" from the given text.
    Allows zero or one space around the colon.
    Returns None if no match is found.
    """
    pattern = r"\[state card\]\s?:\s?(.+)" # Simplified \s{0,1} to \s?
    match = re.search(pattern, text, re.IGNORECASE) # Added IGNORECASE for robustness
    return match.group(1).strip() if match else None

def get_algorithm_type_card(text: str) -> str | None:
    """
    Extracts the content following "[algorithm type card]:" from the given text.
    Allows zero or one space around the colon.
    Returns None if no match is found.
    """
    pattern = r"\[algorithm type card\]\s?:\s?(.+)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else None

def get_problem_card(text: str) -> str | None:
    """
    Extracts the content following "[problem card]:" from the given text.
    Allows zero or one space around the colon.
    Returns None if no match is found.
    """
    pattern = r"\[problem card\]\s?:\s?(.+)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else None

def get_cost_card(text: str) -> str | None:
    """
    Extracts the content following "[cost card]:" from the given text.
    Allows zero or one space around the colon.
    Returns None if no match is found.
    """
    pattern = r"\[cost card\]\s?:\s?(.+)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else None

def extract_xml_think(text: str) -> str:
    """
    Extracts the content within the <reasoning> tag.
    Returns an empty string if tags are not found or an error occurs.
    Note: This split-based parsing is simple but might be fragile with malformed XML.
    """
    try:
        # Ensure both tags are present before splitting to avoid errors if one is missing
        if "<reasoning>" in text and "</reasoning>" in text:
            think_content = text.split("<reasoning>", 1)[-1].split("</reasoning>", 1)[0].strip()
            return think_content
        return ""
    except Exception:
        return ""

def extract_xml_answer(text: str) -> str:
    """
    Extracts content within the <answer> tag and parses specific formats.

    - If the answer format is:
      ***Run heuristic:
      selected heuristic: heuristic_name
      (optional lines)
      ***
      it returns the 'heuristic_name'.
    - If the answer is strictly "***Stop***", it returns "***Stop***".
    - If the <answer> tag contains "Run heuristic:" but "selected heuristic:" is not found,
      it returns an empty string.
    - Otherwise, it returns the trimmed content within the <answer> tag.
    - Returns an empty string if <answer> tags are not found or an error occurs during extraction.
    Note: This split-based parsing is simple but might be fragile with malformed XML.
    """
    try:
        if "<answer>" not in text or "</answer>" not in text:
            return ""
        answer_content = text.split("<answer>", 1)[-1].split("</answer>", 1)[0].strip()
    except Exception:
        return "" # Error during initial split

    if answer_content == "***Stop***":
        return "***Stop***"

    # Using re.IGNORECASE for "Run heuristic" and "selected heuristic" for robustness
    if answer_content.lower().startswith("***run heuristic:"):
        match = re.search(r"selected heuristic:\s*(\S+)", answer_content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            # "Run heuristic:" was present, but "selected heuristic:" was not found.
            return ""
    # If not "***Stop***" and not a "Run heuristic" block that was parsed, return the content.
    return answer_content

def count_xml(text: str) -> float:
    """
    Calculates a score based on the presence and structure of specific XML-like tags
    and patterns. This scoring is specific to an expected format.
    """
    if not isinstance(text, str): # Ensure text is a string
        return 0.0

    count = 0.0
    # Add score if exactly two "***" delimiters are present (often for heuristic block)
    if text.count("***") == 2:
        count += 0.125

    # Score for correctly formatted <reasoning> tags with newlines
    if "<reasoning>\n" in text: # Checks for <reasoning> followed by a newline
        count += 0.125
    if "\n</reasoning>\n" in text: # Checks for </reasoning> preceded and followed by a newline
        count += 0.125

    # Score for correctly formatted <answer> tags with newlines
    if "\n<answer>\n" in text: # Checks for <answer> preceded and followed by a newline
        count += 0.125
        # Penalize for content immediately after the </answer>\n sequence (if any)
        # This part seems to want to penalize trailing content outside the intended structure.
        # Consider if `text.split("\n</answer>\n", 1)` is safer if multiple such blocks could exist.
        # For now, assuming at most one such primary block or focus on the last.
        # Split by the specific sequence and check length of the part after it.
        parts_after_answer_newline_tag = text.split("\n</answer>\n", 1)
        if len(parts_after_answer_newline_tag) > 1:
            count -= len(parts_after_answer_newline_tag[-1].strip()) * 0.001

    # Score for </answer> preceded by a newline, and penalize trailing characters
    elif "\n</answer>" in text: # Handles case where </answer> might not be followed by a newline itself
        count += 0.125
        # Penalize for content immediately after the \n</answer> sequence
        parts_after_answer_tag = text.split("\n</answer>", 1)
        if len(parts_after_answer_tag) > 1:
            # The original `(len(text.split("\n</answer>")[-1]) - 1)` was likely trying to avoid counting
            # a potential single newline character if that was the only thing after.
            # Stripping whitespace before len check simplifies this.
            count -= len(parts_after_answer_tag[-1].strip()) * 0.001
            
    return count