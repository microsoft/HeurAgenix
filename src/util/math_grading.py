"""
PRM800K Grading Logic
Adapted from OpenAI's prm800k repository (grader.py and math_normalize.py)
"""

import re
import sympy
from sympy.parsing import sympy_parser
try:
    from pylatexenc import latex2text
except ImportError:
    latex2text = None

from typing import Optional, Union

# -----------------------------------------------------------------------------
# MATH NORMALIZATION (math_normalize.py)
# -----------------------------------------------------------------------------

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) > 0 and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    if len(substr) < 2:
                        new_str += substr
                        continue
                        
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
                except:
                    new_str += substr
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a_str = string.split("/")[0]
    b_str = string.split("/")[1]
    try:
        a = int(a_str)
        b = int(b_str)
        if string == "{}/{}".format(a, b):
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
    except:
        pass
    return string

def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." 
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    
    if len(string) == 0:
        return string
        
    if string[0] == ".":
        string = "0" + string

    # get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # X/Y changed to \frac{X}{Y}
    string = _fix_a_slash_b(string)

    return string

def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = str(answer).strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
             
        # Normalize boxed content if only boxed content is present
        if answer.startswith("\\boxed{") and answer.endswith("}"):
             answer = answer[7:-1]
             
        return _strip_string(answer)
    except:
        return answer

# -----------------------------------------------------------------------------
# GRADER (grader.py)
# -----------------------------------------------------------------------------

BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"

def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )

def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    if latex2text is None:
        # Fallback if no pylatexenc
        return expr
        
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    
    try:
        expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    except Exception:
        pass

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()

def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False

def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False

def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))

def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False

def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    x = float(x)
    return int(x)

def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step

def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr

def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)

def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True

def _normalize_grading(expr: str) -> str:
    """Normalize answer expressions for grading logic (different from math_normalize)."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    # Units
    for unit in [
        "degree", "cm", "centimeter", "meter", "mile", "second", 
        "minute", "hour", "day", "week", "month", "year", "foot",
        "feet", "inch", "yard",
    ]:
         expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr, flags=re.IGNORECASE)
    
    expr = re.sub(f"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
        
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr

def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems

def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal

def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """
    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0
    PRM800K "grader.grade_answer" logic.
    """
    if given_answer is None or ground_truth is None:
        return False

    # 1. Official MATH normalization check (strict string match)
    ground_truth_normalized_mathd = normalize_answer(ground_truth)
    given_answer_normalized_mathd = normalize_answer(given_answer)

    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    # 2. Advanced Normalization + SymPy check (PRM800K grader logic)
    ground_truth_normalized = _normalize_grading(ground_truth)
    given_normalized = _normalize_grading(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) != len(given_elems):
        return False
        
    # Check parens match if it's a tuple
    if len(ground_truth_elems) > 1:
        if ground_truth_normalized[0] != given_normalized[0] or \
           ground_truth_normalized[-1] != given_normalized[-1]:
           return False

    is_correct = True
    for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
        if _is_frac(ground_truth_elem) and _is_frac(given_elem):
            # if fractions aren't reduced, then shouldn't be marked as correct
            # so, we don't want to allow sympy.simplify in this case
            if ground_truth_elem != given_elem:
                is_correct = False
        elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
            # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
            is_correct = False
        else:
            is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
        
        if not is_correct:
            break

    return is_correct

# Backwards compatibility alias
verify_math_answer = grade_answer
