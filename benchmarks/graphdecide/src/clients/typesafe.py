"""TypeSafe Jev Choice adapter, using the documented HTTP API.

Reference: https://docs.typesafe.ai/api
Dependency: httpx (install into the experiment environment).
One predict call issues one POST: no SDK retries, redirects, or probability repair.
"""

from __future__ import annotations

import math
import os
from typing import Any

import httpx

from .base import (
    BaseDecisionClient,
    ClientCapabilities,
    ClientClosedError,
    ClientTimeoutError,
    ClientTransportError,
    DecisionRequest,
    DecisionResponse,
    InvalidResponseError,
    ProviderHTTPError,
    TokenUsage,
)


class TypeSafeClient(BaseDecisionClient):
    """Official Jev API adapter, with an injectable transport for offline tests.

    Reads TYPESAFE_API_KEY only when api_key is None. The internally constructed
    AsyncClient (and any supplied transport) is owned and closed by this adapter.
    timeout_seconds configures HTTPX connect/read/write/pool timeouts, not a total
    wall-clock deadline. Context-token capacity is deliberately left unknown.
    """

    _CAPABILITIES = ClientCapabilities(supports_probabilities=True, max_options=255)
    _QUESTION_ID = "decision"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "jev-latest",
        base_url: str = "https://api.typesafe.ai/v1",
        timeout_seconds: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        key = os.environ.get("TYPESAFE_API_KEY") if api_key is None else api_key
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Provide api_key or set TYPESAFE_API_KEY")
        key = key.strip()
        if any(char.isspace() for char in key) or not key.isascii():
            raise ValueError("API key must contain ASCII characters without whitespace")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        url = httpx.URL(base_url)
        if (
            url.scheme != "https"
            or not url.host
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("base_url must be an HTTPS URL without credentials/query/fragment")
        self.model = model
        self._closed = False
        self._http = httpx.AsyncClient(
            base_url=str(url).rstrip("/") + "/",
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout_seconds,
            follow_redirects=False,
            transport=transport,
        )

    @property
    def capabilities(self) -> ClientCapabilities:
        return self._CAPABILITIES

    async def predict(self, request: DecisionRequest) -> DecisionResponse:
        if self._closed:
            raise ClientClosedError("TypeSafe client is closed")
        self.validate_request(request)
        payload = {
            "state": request.state,
            "model": self.model,
            "questions": {
                self._QUESTION_ID: {
                    "type": "choice",
                    "instructions": request.question,
                    "criteria": {
                        option.id: option.description for option in request.options
                    },
                }
            },
        }
        try:
            response = await self._http.post("systemone", json=payload)
        except httpx.TimeoutException:
            raise ClientTimeoutError("TypeSafe request timed out") from None
        except httpx.RequestError:
            raise ClientTransportError("TypeSafe transport failed") from None
        if not 200 <= response.status_code < 300:
            raise ProviderHTTPError(
                response.status_code, response.headers.get("retry-after")
            )
        try:
            body = response.json()
        except (ValueError, UnicodeError):
            raise InvalidResponseError(
                "TypeSafe returned invalid JSON", raw_output=response.text
            ) from None
        return self._parse_response(request, body)

    @staticmethod
    def _probability(value: Any, name: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be a finite probability in [0, 1]")
        return float(value)

    @classmethod
    def _parse_response(cls, request: DecisionRequest, body: Any) -> DecisionResponse:
        diagnostic_code = "invalid_response"
        try:
            if not isinstance(body, dict):
                raise ValueError("response must be an object")
            diagnostic_code = "invalid_model"
            model = body.get("model")
            if not isinstance(model, str) or not model.strip():
                raise ValueError("response model is missing or invalid")
            diagnostic_code = "invalid_native_answer"
            answers = body.get("answers")
            if not isinstance(answers, dict) or set(answers) != {cls._QUESTION_ID}:
                raise ValueError("response must contain exactly the requested answer")
            answer = answers[cls._QUESTION_ID]
            if not isinstance(answer, dict) or answer.get("type") != "choice":
                raise ValueError("expected a choice answer")
            ids = [option.id for option in request.options]
            diagnostic_code = "invalid_choice"
            selected = answer.get("choice")
            if not isinstance(selected, str) or selected not in ids:
                raise ValueError("choice is not a supplied option")
            diagnostic_code = "invalid_probabilities"
            raw_probs = answer.get("probabilities")
            if not isinstance(raw_probs, dict) or set(raw_probs) != set(ids):
                raise ValueError("probabilities must cover exactly the supplied options")
            probabilities = {
                option_id: cls._probability(raw_probs[option_id], "option probability")
                for option_id in ids
            }
            # Allow floating-point rounding only; never silently renormalize.
            diagnostic_code = "invalid_probability_sum"
            if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0, abs_tol=1e-6):
                raise ValueError("probabilities must sum to 1")
            diagnostic_code = "choice_probability_mismatch"
            if probabilities[selected] + 1e-6 < max(probabilities.values()):
                raise ValueError("choice does not have the highest probability")
            diagnostic_code = "invalid_confidence"
            confidence = cls._probability(answer.get("confidence"), "confidence")
            diagnostic_code = "invalid_usage"
            raw_usage = body.get("usage")
            if not isinstance(raw_usage, dict):
                raise ValueError("usage must be an object")
            counts = {}
            for name in ("input_tokens", "output_tokens"):
                value = raw_usage.get(name)
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError(f"{name} must be a nonnegative integer")
                counts[name] = value
            return DecisionResponse(
                request_id=request.request_id,
                selected_option_id=selected,
                resolved_model=model,
                probabilities=probabilities,
                probability_kind="native",
                confidence=confidence,
                usage=TokenUsage(**counts),
                raw_output=body,
            )
        except ValueError as exc:
            raise InvalidResponseError(
                str(exc), raw_output=body, diagnostic_code=diagnostic_code
            ) from None

    async def aclose(self) -> None:
        if not self._closed:
            await self._http.aclose()
            self._closed = True