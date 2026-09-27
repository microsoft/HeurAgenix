"""Provider-neutral contracts for choice decisions (Python 3.11+).

Requests contain only model-visible information, never labels or scoring metadata.
Successful calls return DecisionResponse; failures raise DecisionClientError.
Benchmark timing, retries, costs, and ground truth belong to the caller.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Self


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True, slots=True)
class DecisionOption:
    """Stable candidate identity and optional model-visible description."""

    id: str
    description: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.id, "option id")
        if self.description is not None and not isinstance(self.description, str):
            raise ValueError("option description must be a string or None")


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    request_id: str
    state: str | dict[str, Any] | list[Any]
    question: str
    options: tuple[DecisionOption, ...]
    protocol_version: str = "choice-v1"

    def __post_init__(self) -> None:
        _nonempty(self.request_id, "request_id")
        _nonempty(self.question, "question")
        _nonempty(self.protocol_version, "protocol_version")
        if not isinstance(self.state, (str, dict, list)):
            raise ValueError("state must be a string, dict, or list")
        try:
            json.dumps(self.state, allow_nan=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("state must be JSON-serializable with finite numbers") from exc
        # Copy the sequence so later list edits cannot change candidate order.
        object.__setattr__(self, "options", tuple(self.options))
        if len(self.options) < 2:
            raise ValueError("a choice request needs at least two options")
        if not all(isinstance(option, DecisionOption) for option in self.options):
            raise ValueError("options must contain DecisionOption objects")
        ids = [option.id for option in self.options]
        if len(set(ids)) != len(ids):
            raise ValueError("option ids must be unique")


@dataclass(frozen=True, slots=True)
class ClientCapabilities:
    supports_probabilities: bool = False
    max_options: int | None = None
    max_context_tokens: int | None = None
    supports_native_batch: bool = False
    supports_shared_state_questions: bool = False
    supports_thinking_control: bool = False


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """None means unavailable, not zero; never infer hidden reasoning usage."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class DecisionResponse:
    request_id: str
    selected_option_id: str
    resolved_model: str
    probabilities: dict[str, float] | None = None
    probability_kind: Literal["native", "option_logits"] | None = None
    confidence: float | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw_output: dict[str, Any] | None = field(default=None, repr=False)
    status: Literal["success"] = field(default="success", init=False)


class DecisionClientError(RuntimeError):
    """A failed attempt. Callers must retain it rather than dropping the sample."""


class UnsupportedRequestError(DecisionClientError):
    """Request exceeds a client's supported decision protocol or capacity."""


class ClientClosedError(DecisionClientError):
    """A client was used after being closed."""


class ClientTimeoutError(DecisionClientError):
    """The transport timed out; the provider may still have processed the call."""


class ClientTransportError(DecisionClientError):
    """The provider could not be reached or the connection failed."""


class ProviderHTTPError(DecisionClientError):
    def __init__(self, status_code: int, retry_after: str | None = None) -> None:
        # Do not echo request headers, credentials, or provider error bodies.
        super().__init__(f"Provider returned HTTP {status_code}")
        self.status_code = status_code
        self.retry_after = retry_after
        self.retryable = status_code in (408, 429) or 500 <= status_code < 600


class InvalidResponseError(DecisionClientError):
    """Malformed output with allowlisted diagnostics safe for failure artifacts.

    Message and raw_output remain available in memory for compatibility, but must
    never be persisted. Unknown diagnostics become invalid_response/None; invalid
    usage is discarded rather than coerced or partially trusted.
    """

    _DIAGNOSTIC_CODES = frozenset({
        "invalid_response", "missing_answer", "invalid_json", "invalid_choice",
        "incomplete_reasoning", "unexpected_reasoning", "invalid_usage",
        "invalid_finish_reason", "finish_length", "finish_tool_calls",
        "finish_function_call", "finish_content_filter", "finish_error", "finish_abort",
        "invalid_model", "invalid_native_answer", "invalid_probabilities",
        "invalid_probability_sum", "choice_probability_mismatch", "invalid_confidence",
    })
    _FINISH_REASONS = frozenset({
        "stop", "length", "tool_calls", "function_call", "content_filter", "error", "abort",
    })

    def __init__(
        self, message: str, *, raw_output: Any = None,
        diagnostic_code: str = "invalid_response", finish_reason: str | None = None,
        usage: TokenUsage | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.diagnostic_code = (
            diagnostic_code if type(diagnostic_code) is str
            and diagnostic_code in self._DIAGNOSTIC_CODES else "invalid_response"
        )
        self.finish_reason = (
            finish_reason if type(finish_reason) is str
            and finish_reason in self._FINISH_REASONS else None
        )
        self.usage = None
        if type(usage) is TokenUsage:
            values = (usage.input_tokens, usage.output_tokens, usage.reasoning_tokens)
            if all(v is None or (type(v) is int and v >= 0) for v in values):
                self.usage = TokenUsage(*values)


class BaseDecisionClient(ABC):
    """Small asynchronous interface; no implicit retries or answer repair."""

    @property
    @abstractmethod
    def capabilities(self) -> ClientCapabilities:
        """Describe this adapter's implemented features, not vendor promises."""

    def validate_request(self, request: DecisionRequest) -> None:
        if request.protocol_version != "choice-v1":
            raise UnsupportedRequestError("Only choice-v1 is supported")
        limit = self.capabilities.max_options
        if limit is not None and len(request.options) > limit:
            raise UnsupportedRequestError(f"Client supports at most {limit} options")

    @abstractmethod
    async def predict(self, request: DecisionRequest) -> DecisionResponse:
        """Return one decision or raise a typed failure for this attempt."""

    async def predict_many(
        self, requests: Sequence[DecisionRequest]
    ) -> list[DecisionResponse]:
        """Ordered, sequential fallback; stops on the first failure.

        Not native batching. For per-attempt failure logs and resumable runs,
        the benchmark runner should call predict individually.
        """
        for request in requests:
            self.validate_request(request)
        return [await self.predict(request) for request in requests]

    async def initialize(self) -> None:
        """Optional preflight before evaluation; must not send benchmark questions."""

    async def aclose(self) -> None:
        """Override when the adapter owns connections or model resources."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()