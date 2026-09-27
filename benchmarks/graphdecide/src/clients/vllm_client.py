"""Async vLLM chat-completions adapter with explicit chat-template thinking control.

No history, tools, automatic retries, context trimming, or answer repair. HTTPX
implements the OpenAI-compatible REST wire format without a second HTTP SDK.
"""

from __future__ import annotations

import asyncio
import math
import os

import httpx

from ._llm_choice import (
    SYSTEM_MESSAGE, ANSWER_ONLY_SYSTEM_MESSAGE, parse_answer_only,
    parse_choice, prompt_for, reject_unexpected_reasoning,
    token_usage, validate_settings,
)
from .base import (
    BaseDecisionClient, ClientCapabilities, ClientClosedError, ClientTimeoutError,
    ClientTransportError, DecisionRequest, DecisionResponse, InvalidResponseError,
    ProviderHTTPError, TokenUsage,
)


def _response_usage(body: dict) -> TokenUsage:
    usage = body.get("usage")
    if usage is None:
        usage = {}
    if not isinstance(usage, dict):
        raise InvalidResponseError("vLLM usage must be an object", diagnostic_code="invalid_usage")
    details = usage.get("completion_tokens_details")
    if details is None:
        details = {}
    if not isinstance(details, dict):
        raise InvalidResponseError("vLLM completion token details must be an object",
                                   diagnostic_code="invalid_usage")
    return token_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"),
                       details.get("reasoning_tokens"))


class VLLMClient(BaseDecisionClient):
    """Connect to an already-running vLLM server; never launch/download a model.

    ``think`` is sent as ``chat_template_kwargs.enable_thinking``. The served
    model's chat template must support it (for example Qwen3). None leaves the
    server default untouched. An accepted parameter does not prove it was honored.
    """

    def __init__(
        self, *, model: str = "Qwen/Qwen3-8B",
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str | None = None, think: bool | None = False,
        timeout_seconds: float = 180.0, max_tokens: int = 4096,
        temperature: float = 0.0, top_p: float = 1.0, seed: int | None = None,
        top_k: int | None = None, presence_penalty: float | None = None,
        constrain_choices: bool = False,
        output_format: str = "json",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        validate_settings(model, think, timeout_seconds, max_tokens)
        if output_format not in ("json", "answer_only"):
            raise ValueError("output_format must be json or answer_only")
        if type(constrain_choices) is not bool:
            raise ValueError("constrain_choices must be bool")
        if constrain_choices and (think is not False or output_format != "answer_only"):
            raise ValueError("constrain_choices requires think=False and output_format=answer_only")
        self.output_format = output_format
        self.constrain_choices = constrain_choices
        for name, value, upper in (("temperature", temperature, 2), ("top_p", top_p, 1)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= upper
                    or (name == "top_p" and value == 0)):
                raise ValueError(f"Invalid {name}")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        if top_k is not None and (type(top_k) is not int or top_k != -1 and top_k < 1):
            raise ValueError("top_k must be -1 or a positive integer")
        if presence_penalty is not None and (
                type(presence_penalty) not in (int, float) or not math.isfinite(presence_penalty)
                or not -2 <= presence_penalty <= 2):
            raise ValueError("presence_penalty must be finite and in [-2, 2]")
        url = httpx.URL(base_url)
        if (url.scheme not in ("http", "https") or not url.host or url.username
                or url.password or url.query or url.fragment):
            raise ValueError("base_url must be HTTP(S) without credentials/query/fragment")
        key = os.environ.get("VLLM_API_KEY", "") if api_key is None else api_key
        if not isinstance(key, str) or (key and (not key.isascii() or any(c.isspace() for c in key))):
            raise ValueError("API key must be ASCII without whitespace")
        self.model, self.think = model, think
        self.timeout_seconds, self.max_tokens = timeout_seconds, max_tokens
        self.temperature, self.top_p, self.seed = temperature, top_p, seed
        self.top_k, self.presence_penalty = top_k, presence_penalty
        self._closed = False
        self._http = httpx.AsyncClient(
            base_url=str(url).rstrip("/") + "/",
            headers={"Authorization": f"Bearer {key}"} if key else {},
            timeout=timeout_seconds, follow_redirects=False, transport=transport,
        )

    @property
    def capabilities(self) -> ClientCapabilities:
        return ClientCapabilities(supports_thinking_control=True)

    async def predict(self, request: DecisionRequest) -> DecisionResponse:
        if self._closed:
            raise ClientClosedError("vLLM client is closed")
        self.validate_request(request)
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": (
                ANSWER_ONLY_SYSTEM_MESSAGE if self.output_format == "answer_only" else SYSTEM_MESSAGE)},
                         {"role": "user", "content": prompt_for(request)}],
            "temperature": self.temperature, "top_p": self.top_p,
            "max_tokens": self.max_tokens, "stream": False, "n": 1,
        }
        if self.think is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.think}
        if self.seed is not None:
            payload["seed"] = self.seed
        for name in ("top_k", "presence_penalty"):
            if getattr(self, name) is not None:
                payload[name] = getattr(self, name)
        if self.constrain_choices:
            payload["structured_outputs"] = {"choice": [option.id for option in request.options]}
        try:
            async with asyncio.timeout(self.timeout_seconds):
                response = await self._http.post("chat/completions", json=payload)
        except (TimeoutError, httpx.TimeoutException):
            raise ClientTimeoutError("vLLM request timed out") from None
        except httpx.RequestError:
            raise ClientTransportError("vLLM transport failed") from None
        if not 200 <= response.status_code < 300:
            raise ProviderHTTPError(response.status_code, response.headers.get("retry-after"))
        try:
            body = response.json()
        except (ValueError, UnicodeError):
            raise InvalidResponseError("vLLM returned invalid JSON",
                                       diagnostic_code="invalid_json") from None
        return self._parse_response(request, body)

    def _parse_response(self, request: DecisionRequest, body: object) -> DecisionResponse:
        if not isinstance(body, dict):
            raise InvalidResponseError("vLLM response must be an object")
        model, choices = body.get("model"), body.get("choices")
        single_choice = isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict)
        finish_reason = choices[0].get("finish_reason") if single_choice else None
        # Extract metadata independently so parse failures and length stops retain
        # safe counts. Bad usage must not mask the original answer/finish failure.
        usage, usage_error = None, None
        try:
            usage = _response_usage(body)
        except InvalidResponseError as exc:
            usage_error = exc
        try:
            if not isinstance(model, str) or not model.strip():
                raise InvalidResponseError("vLLM response model is missing")
            if not single_choice:
                raise InvalidResponseError("vLLM must return exactly one completion")
            if finish_reason != "stop":
                code = {
                    "length": "finish_length", "tool_calls": "finish_tool_calls",
                    "function_call": "finish_function_call", "content_filter": "finish_content_filter",
                    "error": "finish_error", "abort": "finish_abort",
                }.get(finish_reason, "invalid_finish_reason") if type(finish_reason) is str else "invalid_finish_reason"
                raise InvalidResponseError("vLLM completion did not finish normally", diagnostic_code=code)
            message = choices[0].get("message")
            if (not isinstance(message, dict) or message.get("role") != "assistant"
                    or message.get("tool_calls") or message.get("function_call") or message.get("refusal")):
                raise InvalidResponseError("vLLM must return an assistant answer without tool calls or refusal")
            parser = parse_answer_only if self.output_format == "answer_only" else parse_choice
            selected = parser(message.get("content"), request)
            if usage_error is not None:
                raise usage_error
            if self.think is False:
                reject_unexpected_reasoning(
                    message.get("content"),
                    reasoning=message.get("reasoning_content") or message.get("reasoning"),
                    reasoning_tokens=usage.reasoning_tokens,
                )
        except InvalidResponseError as exc:
            raise InvalidResponseError(
                str(exc), diagnostic_code=exc.diagnostic_code,
                finish_reason=finish_reason, usage=usage,
            ) from None
        return DecisionResponse(
            request_id=request.request_id, selected_option_id=selected, resolved_model=model,
            usage=usage,
            raw_output=body,
        )

    async def aclose(self) -> None:
        if not self._closed:
            await self._http.aclose()
            self._closed = True