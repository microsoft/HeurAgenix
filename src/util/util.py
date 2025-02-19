import ast
import os
import re
import io
import hashlib
import numpy as np
import pandas as pd

def extract(message: str, key: str, sep=None) -> list[str]:
    formats = [
        rf"\*\*\*.*{key}:(.*?)\*\*\*",
        rf" \*\*\*{key}:(.*?)\*\*\*",
        rf"\*\*\*\n{key}:(.*?)\*\*\*"
    ]
    message.replace("\n\n", "\n")
    for format in formats:
        match = re.search(format, message, re.DOTALL)
        value = None
        if match:
            value = match.group(1).strip()
            if value:
                if sep:
                    return value.split(sep)
                else:
                    if value in ["None", "none"]:
                        return None
                    return value
    if sep:
        return []
    else:
        return None

def parse_text_to_dict(text):
    lines = text.split("\n")
    result = {}
    current_key = None
    current_content = []
    for line in lines:
        if len(line) > 0 and line[0] == "-" and ":" in line:
            if current_key:
                result[current_key] = "\n".join(current_content).strip()
            current_key = line[1:].split(":")[0]
            current_content = []
            if len(line.split(":")) > 0:
                current_content = [line.split(":")[1]]
        elif current_key:
            current_content.append(line)
    if current_key:
        result[current_key] = "\n".join(current_content).strip()
    return result

def load_heuristic(heuristic_file:str, problem: str="base", function_name: str=None) -> callable:
    if not "\n" in heuristic_file:
        if not heuristic_file.endswith(".py"):
            # Heuristic name
            heuristic_file += ".py"
        heuristic_path = search_file(heuristic_file, problem)
        assert heuristic_path is not None
        heuristic_code = open(heuristic_path, "r").read()
    else:
        # Heuristic code
        heuristic_code = heuristic_file

    if function_name is None:
        function_name = heuristic_file.split(os.sep)[-1].split(".")[0]
    exec(heuristic_code, globals())
    assert function_name in globals()
    return eval(function_name)

def load_framework_description(component_code: str) -> tuple[str, str]:
    """ Load framework description for the problem from source code, including solution design and operators design."""
    def get_method_source(method_node):
        """Convert the method node to source component_code, ensuring correct indentation."""
        source_lines = ast.unparse(method_node).split('\n')
        indented_source = '\n'.join(['    ' + line for line in source_lines])  # Indent the source component_code
        return indented_source

    tree = ast.parse(component_code)
    solution_str = ""
    operator_str = ""

    # Traverse the AST to find the Solution and Operator classes
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if any(base.id == 'BaseSolution' for base in node.bases if isinstance(base, ast.Name)):
                # Extract Solution class with only __init__ method
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        solution_str += f"class {node.name}:\n"
                        solution_str += f"    \"\"\"{ast.get_docstring(node)}\"\"\"\n" if ast.get_docstring(node) else ""
                        solution_str += get_method_source(item) + "\n"
            elif any(base.id == 'BaseOperator' for base in node.bases if isinstance(base, ast.Name)):
                # Extract Operator class with only __init__ and run methods
                operator_str += f"class {node.name}(BaseOperator):\n"
                operator_str += f"    \"\"\"{ast.get_docstring(node)}\"\"\"\n" if ast.get_docstring(node) else ""
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name in ['__init__', 'run']:
                        operator_str += get_method_source(item) + "\n"

    return solution_str.strip(), operator_str.strip()

def extract_function_with_docstring(code_str, function_name):
    pattern = rf"def {function_name}\(.*?\) -> .*?:\s+\"\"\"(.*?)\"\"\""
    match = re.search(pattern, code_str, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

def filter_dict_to_str(dicts: list[dict], content_threshold: int=None) -> str:
    if isinstance(dicts, dict):
        dicts = [dicts]
    total_dict = {k: v for d in dicts for k, v in d.items()}
    strs = []
    for key, value in total_dict.items():
        if callable(value):
            continue
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if "\n" in str(value):
            key_value_str = str(key) + ":\n" + str(value)
        else:
            key_value_str = str(key) + ":" + str(value)
        if content_threshold is None or len(key_value_str) <= content_threshold:
            strs.append(key_value_str)
    return "\n".join(strs)

def find_key_value(source_dict: dict, key: object) -> object:
    if key in source_dict:
        return source_dict[key]
    else:
        for k, v in source_dict.items():
            if isinstance(v, dict):
                if key in v:
                    return v[key]
    return None

def extract_function_with_short_docstring(code_str, function_name):
    pattern = rf"def {function_name}\(.*?\) -> .*?:\s+\"\"\"(.*?).*Args"
    match = re.search(pattern, code_str, re.DOTALL)
    if match:
        string = match.group(0)
        function_name = string.split("(")[0].strip()
        parameters = string.split("get_state_data_function: callable")[1].split(", **kwargs")[0].strip()
        if parameters[:2] == ", ":
            parameters = parameters[2:]
        introduction = string.split("\"\"\"")[1].split("Args")[0].strip()
        introduction = re.sub(r'\s+', ' ', introduction)
        return f"{function_name}({parameters}): {introduction}"
    else:
        return None

def parse_paper_to_dict(content: str, level=0):
    if level == 0:
        pattern = r'\\section\{(.*?)\}(.*?)((?=\\section)|\Z)'
    elif level == 1:
        pattern = r'\\subsection\{(.*?)\}(.*?)((?=\\subsection)|(?=\\section)|\Z)'
    elif level == 2:
        pattern = r'\\subsubsection\{(.*?)\}(.*?)((?=\\subsubsection)|(?=\\subsection)|(?=\\section)|\Z)'
    else:
        raise ValueError("Unsupported section level")
    sections = re.findall(pattern, content, re.DOTALL)
    section_dict = {}
    for title, body, _ in sections:
        body = body.strip()
        if level < 2:
            sub_dict = parse_paper_to_dict(body, level + 1)
            if sub_dict:
                section_dict[title] = sub_dict
            else:
                section_dict[title] = body
        else:
            section_dict[title] = body
    if level == 0:
        if "\\begin{abstract}" in content:
            section_dict["Abstract"] = content.split("\\begin{abstract}")[-1].split("\\end{abstract}")[0]
        if "\\begin{Abstract}" in content:
            section_dict["Abstract"] = content.split("\\begin{Abstract}")[-1].split("\\end{Abstract}")[0]
        if "\\title{" in content:
            section_dict["Title"] = content.split("\\title{")[-1].split("}")[0]
    return dict(section_dict)

def replace_strings_in_dict(source_dict: dict, replace_value: str="...") -> dict:
    for key in source_dict:
        if isinstance(source_dict[key], str):
            source_dict[key] = replace_value 
        elif isinstance(source_dict[key], dict):
            source_dict[key] = replace_strings_in_dict(source_dict[key])
    return source_dict

def search_file(file_name: str, problem: str="base") -> str:
    def find_file_in_folder(folder_path, file_name):
        return next((os.path.join(root, file_name) for root, dirs, files in os.walk(folder_path) if file_name in files or file_name in dirs), None)

    if os.path.exists(file_name):
        return file_name

    file_path = find_file_in_folder(os.path.join("src", "problems", problem), file_name)
    if file_path:
        return file_path

    if os.getenv("AMLT_DATA_DIR"):
        output_dir = os.getenv("AMLT_DATA_DIR")
    else:
        output_dir = "output"

    file_path = find_file_in_folder(os.path.join(output_dir, problem, "data"), file_name)
    if file_path:
        return file_path

    file_path = find_file_in_folder(os.path.join(output_dir, problem, "heuristics"), file_name)
    if file_path:
        return file_path

    file_path = find_file_in_folder(os.path.join(output_dir, problem), file_name)
    if file_path:
        return file_path
    
    return None

def df_to_str(df: pd.DataFrame) -> str:
    return df.to_csv(sep="\t", index=False).replace("\r\n", "\n").strip()

def str_to_df(string: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(string), sep="\t")

def sanitize_function_name(name: str, id_str: str="None"):
    s1 = re.sub('(.)([A-Z][a-z]+)', r"\1_\2", name)
    sanitized_name = re.sub('([a-z0-9])([A-Z])', r"\1_\2", s1).lower()

    # Replace spaces with underscores
    sanitized_name = sanitized_name.replace(" ", "_").replace("__", "_")

    # Remove invalid characters
    sanitized_name = "".join(char for char in sanitized_name if char.isalnum() or char == '_')

    # Ensure it doesn't start with a digit
    if sanitized_name and sanitized_name[0].isdigit():
        sanitized_name = "_" + sanitized_name

    suffix_str = hashlib.sha256(id_str.encode()).hexdigest()[:4]
    # Add uuid to avoid duplicated name
    sanitized_name = sanitized_name + "_" + suffix_str

    return sanitized_name