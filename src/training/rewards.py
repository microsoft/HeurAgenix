import re
from utils import extract_xml_answer, ALGORITHM_NAMES, count_xml, extract_xml_think, get_state_card, get_algorithm_type_card, get_problem_card, get_cost_card

import spacy
from langdetect import detect, DetectorFactory

# Dictionary categorizing algorithm names
alg_dic = {
    'refinement':[
        "simulated_annealing_e625",
        "_2opt_89aa",
        "_3opt_e75b",
        "three_opt_e8d7",
        "two_opt_0554",
        "node_shift_between_routes_7b8a",
    ],
    'exploration':[
        "nearest_insertion_c1f0",
        "nearest_neighbor_f91d",
        "random_80a0",
        "farthest_insertion_b6d3",
        "greedy_randomized_adaptive_search_procedure_grasp_5a6a",
        "ant_colony_d4f7",
        "cheapest_insertion_605f",
        "random_successive_insertion_57b4",
        "insertion_heuristics_050b",
        "greedy_algorithm_3ca7",
        "k_nearest_neighbors_insertion_9e8b",
        "random_pairwise_insertion_7493",
        "farthest_insertion_4e1d",
        "greedy_f4c4",
        "min_cost_insertion_7bfa",
        "nearest_neighbor_99ba",
        "petal_algorithm_b384",
        "random_bfdc",
        "saving_algorithm_710e",
        "variable_neighborhood_search_614b",
    ]
}

# Set a fixed random seed for langdetect to ensure detection stability
DetectorFactory.seed = 0

# Need to validate multiple cards simultaneously
def cards_reward_func(completions, answer, **kwargs) -> list[float]:
    # Extract full content from each completion
    responses = [completion[0]['content'] for completion in completions]

    rewards = []
    for idx, response in enumerate(responses):
        tem_reward = 0.0
        info = answer[idx] # Ground truth information for the current response

        # Problem card evaluation
        tem_problem_card = get_problem_card(response)
        if tem_problem_card is None:
            tem_reward -= 2
        elif tem_problem_card not in ['tsp','cvrp']:
            tem_reward -= 2
        else:
            if tem_problem_card == info["card_problem"][0]:
                tem_reward += 1
            else:
                tem_reward -= 1

        # State card evaluation
        tem_state_card = get_state_card(response)
        if tem_state_card is None:
            tem_reward -= 2
        elif tem_state_card not in ['partially visited', 'fully visited', 'unvisited']:
            tem_reward -= 2
        else:
            if tem_state_card == info["card_state"][0]:
                tem_reward += 1
            else:
                tem_reward -= 1

        # Algorithm type card evaluation
        tem_algorithm_type_card = get_algorithm_type_card(response)
        if tem_algorithm_type_card is None:
            tem_reward -= 2 # Penalize if algorithm_type_card is None
        elif tem_algorithm_type_card not in ['refinement','exploration']:
            tem_reward -= 2
        else:
            gen_answer = extract_xml_answer(response)
            if tem_algorithm_type_card == info["card_alg_type"][0]:
                tem_reward += 1 # Reward if consistent with the ground truth answer's algorithm type
            elif gen_answer in alg_dic.get(tem_algorithm_type_card, []):
                # Reward if self-consistent (generated algorithm matches the category of the generated type card)
                tem_reward += 0.5
            else:
                tem_reward -= 1

        # Cost card evaluation
        tem_cost_card = get_cost_card(response)
        if tem_cost_card is None:
            tem_reward -= 2
        elif tem_cost_card not in ['None', 'low cost', 'normal cost', 'high cost']:
            tem_reward -= 2
        else:
            # No score change if either generated or actual cost card is 'None'
            if tem_cost_card == 'None' or info["card_cost"][0] == 'None':
                tem_reward -= 0 # Explicitly no change
            elif tem_cost_card == info["card_cost"][0]:
                tem_reward += 1 # Reward if consistent with the answer
            else:
                tem_reward -= 1

        rewards.append(tem_reward)

    return rewards


def language_consistency_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    This reward function checks if the 'think' part of each completion is entirely in English.
    
    For each 'think' text:
        - If the language can be detected and is English, a positive reward (1.0) is given;
        - Otherwise, a negative reward (-1.0) is given.
    
    Args:
        prompts: The prompts sent to the model.
        completions: A list containing model outputs, where each element is a list of generated sequences,
                     and each sequence (dictionary) should have a 'content' field.
        answer: The ground truth answer.
        **kwargs: Additional keyword arguments.
    
    Returns:
        A list of floats, corresponding to the reward score for each input.
    """
    # Extract full content from each completion
    responses = [completion[0]['content'] for completion in completions]
    # Use existing tools to extract the thinking part
    extracted_thinks = [extract_xml_think(r) for r in responses]
    
    rewards = []
    for think_text in extracted_thinks:
        # If the 'think' part is empty, penalize directly
        if not think_text:
            rewards.append(-1.0)
            continue
        
        try:
            lang = detect(think_text)
            # If detected language is English, give positive reward, otherwise penalize
            if lang == 'en':
                rewards.append(1.0)
            else:
                rewards.append(-1.0)
        except Exception:
            # If an exception occurs during detection, also consider it as not meeting requirements
            rewards.append(-1.0)
    
    return rewards

# Load English SpaCy model (medium size)
# Ensure you have this model downloaded: python -m spacy download en_core_web_md
nlp = spacy.load('en_core_web_md')

def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculates the semantic similarity between two texts using SpaCy word vectors.
    Returns the cosine similarity score.
    """
    if not text1 or not text2: # Handle empty strings to avoid SpaCy errors / NaN
        return 0.0
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    # Handle cases where documents might not have vectors (e.g., only OOV tokens or very short)
    if not doc1.has_vector or not doc2.has_vector or doc1.vector_norm == 0 or doc2.vector_norm == 0:
        return 0.0
    return doc1.similarity(doc2)

def think_answer_consistency_reward_func(completions, similarity_threshold=0.5, **kwargs) -> list[float]:
    """
    This reward function calculates the semantic similarity between the extracted
    answer and the thinking process from each completion. It returns the raw similarity score.

    Args:
        completions: A list containing model outputs, where each element is a list of generated sequences,
                     and each sequence (dictionary) should have a 'content' field.
        similarity_threshold: (Currently unused in the returned score but kept for potential future use)
                              Threshold for judging consistency.
        **kwargs: Additional keyword arguments.
    
    Returns:
        A list of floats, corresponding to the raw semantic similarity score for each input.
    """
    # Extract full content from each completion
    responses = [completion[0]['content'] for completion in completions]
    # Use existing tools to extract the answer part
    extracted_answers = [extract_xml_answer(r) for r in responses]
    # Use existing tools to extract the thinking part
    extracted_thinks = [extract_xml_think(r) for r in responses]
    
    similarities = []
    for ans, think_text in zip(extracted_answers, extracted_thinks):
        # If the answer or thinking part is missing, similarity will be low (or 0.0 based on compute_semantic_similarity)
        similarity = compute_semantic_similarity(ans if ans else "", think_text if think_text else "")
        similarities.append(similarity)
    
    return similarities


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    # Extract response text from each completion
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    # print('-'*20,
    #       f"\nAnswer:\n{answer}",
    #       f"\nResponse:\n{responses}",
    #       f"\nExtracted:\n{extracted_responses}")
    
    rewards = []
    for idx, extracted in enumerate(extracted_responses):
        info = answer[idx] # Ground truth for the current response
        reward = 0.0 # Default reward
        
        # Judgment by priority: correct_answer > illegal > disabled_algorithms > positive_samples > negative_samples > else
        if extracted == info["correct_answer"]: # Note: info["correct_answer"] might be a list in original data, ensure comparison is correct
            reward = 1.0 # Max reward for perfect match
        
        elif extracted in info.get("illegal", []):
            reward = -1.0
            
        elif extracted in info.get("disabled_algorithms", []):
            reward = -2.0 # Higher penalty for using disabled algorithms
        else:
            # positive_samples: max +1, linear decay
            pos_list = info.get("positive_samples", [])
            if extracted in pos_list:
                L = len(pos_list)
                k = pos_list.index(extracted) # Index of the matched positive sample
                # Reward decreases as the index k increases (less preferred positive samples get lower scores)
                reward = 1.0 * ((L - 1 - k) / (L - 1) if L > 1 else 1.0)
            
            # negative_samples: max –1, linear decay
            elif extracted in info.get("negative_samples", []):
                neg_list = info["negative_samples"]
                L = len(neg_list)
                k = neg_list.index(extracted) # Index of the matched negative sample
                # Penalty increases (becomes more negative) as index k increases
                reward = -1.0 * ((k) / (L - 1) if L > 1 else 1.0)
            
        
        rewards.append(reward)
    
    return rewards


def algorithm_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    rewards = []
    for r_text in extracted_responses:
        if r_text in ALGORITHM_NAMES: # ALGORITHM_NAMES should be a list or set of valid algorithm names from utils
            rewards.append(0.5)
        else:
            rewards.append(-1.0)
    return rewards

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Strict Format Reward Function:
    The response must strictly match a specific XML-like format.
    Constraints are slightly relaxed (e.g., optional whitespace) to allow some flexibility.
    """
    pattern = re.compile(
        r"""
        ^\s* # Optional whitespace at the beginning of the line
        (?:'{3}\s*\n)?                          # Optional starting triple quotes ''' and newline
        \[problem\ card\]\s*:\s*[^\n]+\n
        \[state\ card\]\s*:\s*[^\n]+\n
        \[algorithm\ type\ card\]\s*:\s*[^\n]+\n
        (?:\[cost\ card\]\s*:\s*[^\n]+\n)?\s* # Optional cost card, followed by optional newline/whitespace
        <reasoning>[\s\S]*?</reasoning>\s*\n
        <answer>\s*\n
        \*\*\*Run\ heuristic\s*:\s*\n
        selected\ heuristic\s*:\s*[^\n]+\n
        (?:hyper\ parameter\s*\(optional\)\s*:\s*[^\n]+\n)?
        \*\*\*\s*\n
        </answer>
        (?:\s*\n'{3})?                          # Optional ending triple quotes and preceding newline/whitespace
        \s*$                                    # Optional whitespace at the end of the line
        """,
        re.MULTILINE | re.VERBOSE
    )

    responses = [completion[0]["content"] for completion in completions]
    # Ensure response is a string, handle None if content could be missing
    matches = [re.fullmatch(pattern, r) if isinstance(r, str) else None for r in responses]
    return [1.0 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Soft Format Reward Function:
    Gives a graded score based on the presence of key structural elements.
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for response in responses:
        score = 0.0
        if not isinstance(response, str): # Handle cases where response might not be a string
            rewards.append(0.0)
            continue
            
        # Check reasoning part (0.1 points)
        if re.search(r"<reasoning>[\s\S]*?</reasoning>", response, re.DOTALL): # Allow content on same line
            score += 0.1
        
        # Check answer part (0.1 points)
        if re.search(r"<answer>[\s\S]*?</answer>", response, re.DOTALL): # Allow content on same line
            score += 0.1
        
        # Check "Run heuristic" block with selected heuristic and three asterisks (0.2 points)
        # Made the pattern more robust to internal newlines if any.
        if re.search(r"\*\*\*Run\ heuristic\s*:\s*\n\s*selected\ heuristic\s*:\s*[^\n]+(?:[\s\S]*?)\*\*\*", response, re.DOTALL):
            score += 0.2
        
        # Ensure score does not exceed a theoretical max if more checks are added
        # For now, max is 0.1+0.1+0.2 = 0.4. If a 1.0 scale is desired, normalize later.
        rewards.append(score)
    
    return rewards

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """
    Counts specific XML tags or patterns. Assumes count_xml from utils returns a numerical score.
    """
    contents = [completion[0]["content"] for completion in completions]
    return [float(count_xml(c)) if isinstance(c, str) else 0.0 for c in contents]
