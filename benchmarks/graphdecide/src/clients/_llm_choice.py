"""Shared, label-free text protocol for generative choice adapters."""

from __future__ import annotations

import json
import math
import re

from .base import DecisionRequest, InvalidResponseError, TokenUsage


SYSTEM_MESSAGE = (
    "Solve the supplied choice problem using only its state, question, and options. "
    "Do not use tools or external information. Return your final answer as exactly "
    'one JSON object with one key: {"choice": "<option id>"}. '
    "Use an option ID verbatim. Do not include explanations or additional keys in the final answer."
)


ANSWER_ONLY_SYSTEM_MESSAGE = (
    "Solve the supplied choice problem using only its state, question, and options. "
    "Do not use tools or external information. Return your final answer as exactly "
    "one option ID, such as yes or no. "
    "Do not include JSON, quotes, formatting, punctuation, explanations, or additional text in the final answer."
)


def parse_answer_only(content: object, request: DecisionRequest) -> str:
    """Accept one complete ID, not an answer extracted from prose or JSON.

    Normalize boundary whitespace and ASCII yes/no case only. As for JSON mode,
    a complete leading think block may precede the final answer; the adapter
    independently rejects nonempty reasoning for no-think requests.
    """
    if not isinstance(content, str) or not content.strip():
        raise InvalidResponseError("Missing final choice text", diagnostic_code="missing_answer")
    text = content.strip()
    if text.startswith("<think>"):
        end = text.find("</think>")
        if end < 0 or "<think>" in text[len("<think>"):end]:
            raise InvalidResponseError("Incomplete or nested thinking block",
                                       diagnostic_code="incomplete_reasoning")
        text = text[end + len("</think>"):].strip()
    if not text:
        raise InvalidResponseError("Missing final choice text", diagnostic_code="missing_answer")
    options = {o.id for o in request.options}
    if options == {"yes", "no"} and text.isascii():
        text = text.lower()
    if text not in options:
        raise InvalidResponseError("Final answer must be exactly one option ID",
                                   diagnostic_code="invalid_choice")
    return text


def prompt_for(request: DecisionRequest) -> str:
    """Exclude evaluator identifiers, labels, source names, and protocol metadata."""
    return json.dumps({
        "state": request.state,
        "question": request.question,
        "options": [{"id": o.id, "description": o.description} for o in request.options],
    }, ensure_ascii=False, allow_nan=False)


def parse_choice(content: object, request: DecisionRequest) -> str:
    """Parse only a complete final answer, never search reasoning for an option.

    Allow one leading, explicitly delimited thinking block and an optional JSON
    fence. Incomplete thinking, prose, duplicate keys and extra JSON are failures.
    """
    if not isinstance(content, str) or not content.strip():
        raise InvalidResponseError("Missing final choice text", diagnostic_code="missing_answer")
    text = content.strip()
    if text.startswith("<think>"):
        end = text.find("</think>")
        if end < 0 or "<think>" in text[len("<think>"):end]:
            raise InvalidResponseError("Incomplete or nested thinking block",
                                       diagnostic_code="incomplete_reasoning")
        text = text[end + len("</think>"):].strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if not text:
        raise InvalidResponseError("Missing final choice text", diagnostic_code="missing_answer")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate key")
            result[key] = value
        return result

    try:
        answer = json.loads(text, object_pairs_hook=unique_object)
    except (ValueError, RecursionError):
        raise InvalidResponseError("Final answer must be a single choice JSON object",
                                   diagnostic_code="invalid_json") from None
    if (not isinstance(answer, dict) or set(answer) != {"choice"}
            or not isinstance(answer["choice"], str)
            or answer["choice"] not in {o.id for o in request.options}):
        raise InvalidResponseError("Final choice must match exactly one supplied option ID",
                                   diagnostic_code="invalid_choice")
    return answer["choice"]


def token_usage(input_tokens=None, output_tokens=None, reasoning_tokens=None) -> TokenUsage:
    values = (input_tokens, output_tokens, reasoning_tokens)
    if any(v is not None and (type(v) is not int or v < 0) for v in values):
        raise InvalidResponseError("Token counts must be nonnegative integers or null",
                                   diagnostic_code="invalid_usage")
    return TokenUsage(*values)


def reject_unexpected_reasoning(content: object, *, reasoning=None, reasoning_tokens=None) -> None:
    """Fail a requested no-think call when the response visibly contradicts it.

    Missing reasoning metadata is not proof that a server honored the setting.
    An empty think block is allowed because some templates always emit one.
    """
    block = re.match(r"\s*<think>(.*?)</think>", content, re.DOTALL) if isinstance(content, str) else None
    if (reasoning or (type(reasoning_tokens) is int and reasoning_tokens > 0)
            or (block and block.group(1).strip())):
        raise InvalidResponseError("Provider returned reasoning despite a no-think request",
                       diagnostic_code="unexpected_reasoning")


def validate_settings(model: str, think: bool | None, timeout_seconds: float, max_tokens: int) -> None:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")
    if think is not None and type(think) is not bool:
        raise ValueError("think must be bool or None")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be finite and positive")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")