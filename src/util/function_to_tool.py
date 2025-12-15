import ast
import json
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

BASIC_TYPE_MAP = {
    "int": "integer",
    "float": "number",
    "str": "string",
    "bool": "boolean",
    "dict": "object",
    "Dict": "object",
    "list": "array",
    "List": "array",
}

def parse_docstring(doc: Optional[str]) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    if not doc:
        return "", {}

    doc = textwrap.dedent(doc).strip()
    lines = doc.splitlines()

    desc_lines = []
    i = 0
    while i < len(lines) and not re.match(r"^\s*Args:\s*$", lines[i]):
        desc_lines.append(lines[i])
        i += 1
    description = "\n".join(desc_lines).strip()

    params_info: Dict[str, Dict[str, Any]] = {}

    if i < len(lines) and re.match(r"^\s*Args:\s*$", lines[i]):
        i += 1
        current_param = None
        while i < len(lines) and not re.match(r"^\s*Returns:\s*$", lines[i]):
            line = lines[i]

            m = re.match(r"^\s*([a-zA-Z_]\w*)\s*\(([^)]+)\)\s*:\s*(.*)\s*$", line)
            if m:
                name = m.group(1)
                typ = m.group(2).strip()
                desc = m.group(3).strip()
                params_info[name] = {"type": typ, "description": desc}
                current_param = name
            else:
                if current_param and (re.match(r"^\s{2,}", line) or re.match(r"^\s*-\s", line)):
                    extra = line.strip()
                    if extra:
                        params_info[current_param]["description"] += "\n" + extra
                else:
                    pass
            i += 1

    return description, params_info


def ast_annotation_to_schema(node: Optional[ast.AST]) -> Optional[Dict[str, Any]]:
    if node is None:
        return None

    if isinstance(node, ast.Name):
        t = node.id
        if t in BASIC_TYPE_MAP:
            return {"type": BASIC_TYPE_MAP[t]}
        return None

    # list[int], List[int], Dict[str, int]
    if isinstance(node, ast.Subscript):
        container = None
        if isinstance(node.value, ast.Name):
            container = node.value.id
        elif isinstance(node.value, ast.Attribute):
            container = node.value.attr

        if container in ("list", "List"):
            items_schema = {"type": "string"}
            inner = node.slice
            inner_node = None
            if isinstance(inner, ast.Index):
                inner_node = inner.value
            else:
                inner_node = inner
            inner_schema = ast_annotation_to_schema(inner_node)
            if inner_schema and "type" in inner_schema:
                items_schema = inner_schema
            return {"type": "array", "items": items_schema}

        if container in ("dict", "Dict"):
            return {"type": "object"}

    if isinstance(node, ast.Attribute):
        return {"type": "object"}

    return None


def doc_type_to_schema_type(doc_type: str) -> Optional[Dict[str, Any]]:
    t = doc_type.strip()
    if t in BASIC_TYPE_MAP:
        return {"type": BASIC_TYPE_MAP[t]}

    # list[int], list[str]
    m = re.match(r"^(?:list|List)\s*\[\s*([a-zA-Z_][\w]*)\s*\]$", t)
    if m:
        inner = m.group(1)
        inner_type = BASIC_TYPE_MAP.get(inner, "string")
        return {"type": "array", "items": {"type": inner_type}}

    # numpy.ndarray -> array
    if "ndarray" in t or "numpy" in t:
        return {"type": "array"}

    # dict/Dict/Mapping
    if t.lower() == "mapping" or "dict" in t or "Dict" in t:
        return {"type": "object"}

    return None


def get_param_default(fn_node: ast.FunctionDef, name: str) -> Any:
    args = fn_node.args
    all_args = [a.arg for a in args.args]
    defaults = args.defaults
    if not defaults:
        return None
    offset = len(all_args) - len(defaults)
    if name in all_args:
        idx = all_args.index(name)
        d_idx = idx - offset
        if 0 <= d_idx < len(defaults):
            node = defaults[d_idx]
            if isinstance(node, ast.Constant):
                return node.value
    return None


def build_param_schema(
    ann_schema: Optional[Dict[str, Any]],
    doc_schema: Optional[Dict[str, Any]],
    doc_desc: Optional[str],
    default_val: Any,
) -> Dict[str, Any]:
    schema: Dict[str, Any] = {}

    # Get type: from notes -> from doc string -> string
    if ann_schema and "type" in ann_schema:
        schema["type"] = ann_schema["type"]
        if ann_schema.get("type") == "array" and "items" in ann_schema:
            schema["items"] = ann_schema["items"]
    elif doc_schema and "type" in doc_schema:
        schema["type"] = doc_schema["type"]
        if doc_schema.get("type") == "array" and "items" in doc_schema:
            schema["items"] = doc_schema["items"]
    else:
        schema["type"] = "string"

    if doc_desc:
        schema["description"] = doc_desc

    # Default value
    if isinstance(default_val, (int, float, str, bool)):
        schema["default"] = default_val

    return schema


def convert_function_to_tool(function_name: str, code: str=None, code_file: str=None) -> List[Dict[str, Any]]:
    if not code and code_file:
        code = open(code_file, "r", encoding="utf-8").read()
    mod = ast.parse(code)

    target_nodes = [n for n in mod.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function_name]
    if len(target_nodes) != 1:
        return None
    node = target_nodes[0]
    SKIP_PARAMS = {"problem_state", "algorithm_data", "args", "kwargs"}

    # Get args and docstring
    arg_nodes = node.args.args
    doc = ast.get_docstring(node)
    description, doc_params = parse_docstring(doc)

    properties: Dict[str, Any] = {}

    for a in arg_nodes:
        pname = a.arg
        if pname in SKIP_PARAMS:
            continue

        # notes -> schema
        ann_schema = ast_annotation_to_schema(a.annotation)
        # docstring -> schema
        doc_schema = None
        doc_desc = None
        if pname in doc_params:
            doc_schema = doc_type_to_schema_type(doc_params[pname].get("type", ""))
            doc_desc = doc_params[pname].get("description", "")

        default_val = get_param_default(node, pname)
        prop_schema = build_param_schema(ann_schema, doc_schema, doc_desc, default_val)
        properties[pname] = prop_schema

    tool = {
        "type": "function",
        "function": {
            "name": node.name,
            "description": description if description else f"Heuristic function {node.name}.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "additionalProperties": False,
            },
        },
    }

    return tool

