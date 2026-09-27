"""Isolated single-question Copilot SDK adapter for GPT and other Copilot models.

The optional SDK is imported lazily. No game-agent code, device login, credentials
cache, tools, workspace instructions, or cross-question history is reused.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import os
from tempfile import TemporaryDirectory

from ._llm_choice import (
    SYSTEM_MESSAGE, ANSWER_ONLY_SYSTEM_MESSAGE, parse_answer_only,
    parse_choice, prompt_for, reject_unexpected_reasoning,
    token_usage, validate_settings,
)
from .base import (
    BaseDecisionClient, ClientCapabilities, ClientClosedError, ClientTimeoutError,
    ClientTransportError, DecisionClientError, DecisionRequest, DecisionResponse,
    InvalidResponseError, UnsupportedRequestError,
)


def _sdk_types():
    try:
        from copilot import CopilotClient
        from copilot.session import (
            ModelCapabilitiesOverride, ModelLimitsOverride, PermissionDecisionUserNotAvailable,
        )
    except ImportError:
        raise ImportError("Install the optional Copilot dependency: pip install -e '.[copilot]'") from None
    return CopilotClient, ModelCapabilitiesOverride, ModelLimitsOverride, PermissionDecisionUserNotAvailable


class GitHubCopilotClient(BaseDecisionClient):
    """One fresh, tool-free session per prediction, using the current asyncio loop.

    Authentication uses an explicit token, COPILOT_GITHUB_TOKEN, or an existing
    SDK/CLI login. Login is never started interactively by this adapter.
    ``think=None`` keeps the model default. ``think=True`` selects a supported
    reasoning effort. ``think=False`` is rejected for reasoning-capable models:
    SDK 1.0.14 has no native 'none' effort. 'low' is not no-thinking.
    ``output_format`` selects JSON (default) or the shared answer-only protocol;
    it does not change reasoning settings.

    One send per question, no adapter retries. The SDK may retry internally; an
    observed retry/multiple model calls is rejected rather than scored as one.
    Always close the client to release its runtime and temporary session files.
    """

    def __init__(
        self, *, model: str = "gpt-4o", github_token: str | None = None,
        think: bool | None = None, reasoning_effort: str | None = None,
        timeout_seconds: float = 180.0, max_tokens: int = 4096,
        output_format: str = "json",
    ) -> None:
        validate_settings(model, think, timeout_seconds, max_tokens)
        if output_format not in ("json", "answer_only"):
            raise ValueError("output_format must be json or answer_only")
        self.output_format = output_format
        if reasoning_effort is not None and reasoning_effort not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError("Unsupported SDK reasoning_effort")
        if think is False and reasoning_effort is not None:
            raise ValueError("reasoning_effort conflicts with think=False")
        token = os.environ.get("COPILOT_GITHUB_TOKEN") if github_token is None else github_token
        if token is not None and (not isinstance(token, str) or not token
                                  or not token.isascii() or any(c.isspace() for c in token)):
            raise ValueError("GitHub token must be nonempty ASCII without whitespace")
        self.model, self.think, self.reasoning_effort = model, think, reasoning_effort
        self.timeout_seconds, self.max_tokens = timeout_seconds, max_tokens
        self._token = token
        self._runtime = None
        self._started = False
        self._directory = None
        self._model_info = None
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def capabilities(self) -> ClientCapabilities:
        # Specific mode support is checked against the account's model metadata.
        return ClientCapabilities(supports_thinking_control=True)

    async def _ensure_runtime(self) -> None:
        if self._model_info is not None:
            return
        client_type, capabilities_type, limits_type, deny_type = _sdk_types()
        if self._runtime is None:
            self._directory = TemporaryDirectory(prefix="jevgraphbench-copilot-")
            self._limits = capabilities_type(limits=limits_type(max_output_tokens=self.max_tokens))
            self._deny_permission = lambda *_: deny_type()
            self._runtime = client_type(
                github_token=self._token, use_logged_in_user=self._token is None,
                mode="empty", base_directory=self._directory.name, log_level="error",
            )
        if not self._started:
            await self._runtime.start()
            self._started = True
        auth = await self._runtime.get_auth_status()
        if not auth.isAuthenticated:
            raise ClientTransportError("Copilot authentication required; log in separately or set COPILOT_GITHUB_TOKEN")
        models = await self._runtime.list_models()
        self._model_info = next((m for m in models if m.id == self.model), None)
        if self._model_info is None:
            raise UnsupportedRequestError("Requested Copilot model is unavailable for this account")

    def _effort(self) -> str | None:
        info = self._model_info
        supported = getattr(info, "supported_reasoning_efforts", None) or []
        supports = getattr(getattr(info, "capabilities", None), "supports", None)
        reasoning = bool(supported or getattr(supports, "reasoning_effort", False)
                         or getattr(info, "default_reasoning_effort", None))
        if self.think is False and reasoning:
            raise UnsupportedRequestError("Copilot SDK cannot disable reasoning for this model; low is not no-think")
        effort = self.reasoning_effort
        if self.think is True and effort is None:
            effort = getattr(info, "default_reasoning_effort", None) or "medium"
        if effort is not None and effort not in supported:
            raise UnsupportedRequestError("Requested reasoning effort is not advertised by this Copilot model")
        return effort

    async def initialize(self) -> None:
        """Fail once for unavailable models/auth/modes, before any scored attempt."""
        async with self._lock:
            if self._closed:
                raise ClientClosedError("Copilot client is closed")
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    await self._ensure_runtime()
                    self._effort()
            except TimeoutError:
                raise ClientTimeoutError("Copilot initialization timed out") from None
            except (DecisionClientError, ImportError):
                raise
            except Exception:
                raise ClientTransportError("Copilot initialization failed") from None

    async def predict(self, request: DecisionRequest) -> DecisionResponse:
        if self._closed:
            raise ClientClosedError("Copilot client is closed")
        self.validate_request(request)
        async with self._lock:
            if self._closed:
                raise ClientClosedError("Copilot client is closed")
            session = None
            completed = False
            usage_events = []
            retried = False

            def on_event(event):
                nonlocal retried
                kind = getattr(event.type, "value", event.type)
                if kind == "assistant.usage":
                    usage_events.append(event.data)
                elif kind == "assistant.turn_retry":
                    retried = True

            try:
                async with asyncio.timeout(self.timeout_seconds):
                    await self._ensure_runtime()
                    effort = self._effort()
                    session = await self._runtime.create_session(
                        model=self.model, reasoning_effort=effort,
                        tools=[], available_tools=[], tool_search={"enabled": False},
                        on_permission_request=self._deny_permission,
                        system_message={"mode": "replace", "content": (
                            ANSWER_ONLY_SYSTEM_MESSAGE if self.output_format == "answer_only" else SYSTEM_MESSAGE)},
                        working_directory=self._directory.name,
                        model_capabilities=self._limits, streaming=False,
                        memory={"enabled": False}, infinite_sessions={"enabled": False},
                        enable_config_discovery=False, skip_custom_instructions=True,
                        enable_on_demand_instruction_discovery=False,
                        enable_skills=False, skill_directories=[], plugin_directories=[],
                        instruction_directories=[], enable_file_hooks=False,
                        enable_host_git_operations=False, enable_session_store=False,
                        enable_file_change_tracking=False, enable_session_telemetry=False,
                        skip_embedding_retrieval=True, mcp_servers={}, enable_mcp_apps=False,
                        request_extensions=False, request_canvas_renderer=False,
                        on_auto_mode_switch_request=lambda *_: "no", on_event=on_event,
                    )
                    event = await session.send_and_wait(prompt_for(request), timeout=self.timeout_seconds)
                    if retried or len(usage_events) > 1:
                        raise InvalidResponseError("Copilot runtime retried or made multiple model calls")
                    if event is None or getattr(event.type, "value", event.type) != "assistant.message":
                        raise InvalidResponseError("Copilot returned no final assistant message")
                    data = event.data
                    if getattr(data, "tool_requests", None):
                        raise InvalidResponseError("Copilot returned unexpected tool requests")
                    usage = usage_events[0] if usage_events else None
                    finish = getattr(usage, "finish_reason", None)
                    if finish not in (None, "stop", "end_turn"):
                        raise InvalidResponseError("Copilot completion did not finish normally")
                    if getattr(usage, "content_filter_triggered", False):
                        raise InvalidResponseError("Copilot completion was filtered")
                    usage_model, message_model = getattr(usage, "model", None), getattr(data, "model", None)
                    if usage_model and message_model and usage_model != message_model:
                        raise InvalidResponseError("Copilot reported inconsistent response models")
                    model = usage_model or message_model
                    if not isinstance(model, str) or not model.strip():
                        raise InvalidResponseError("Copilot did not report the response model")
                    actual_effort = getattr(usage, "reasoning_effort", None)
                    if effort is not None and actual_effort is not None and effort != actual_effort:
                        raise InvalidResponseError("Copilot reported a different reasoning effort than requested")
                    if self.think is False:
                        reject_unexpected_reasoning(
                            data.content,
                            reasoning=(getattr(data, "reasoning_text", None)
                                       or getattr(data, "reasoning_blocks", None)
                                       or actual_effort),
                            reasoning_tokens=getattr(usage, "reasoning_tokens", None),
                        )
                    parser = parse_answer_only if self.output_format == "answer_only" else parse_choice
                    selected = parser(data.content, request)
                    result = DecisionResponse(
                        request_id=request.request_id, selected_option_id=selected, resolved_model=model,
                        usage=token_usage(getattr(usage, "input_tokens", None),
                                          getattr(usage, "output_tokens", None),
                                          getattr(usage, "reasoning_tokens", None)),
                        raw_output={"content": data.content, "model": model,
                                    "finish_reason": finish, "reasoning_effort": getattr(usage, "reasoning_effort", None)},
                    )
                    completed = True
                    return result
            except TimeoutError:
                raise ClientTimeoutError("Copilot operation timed out") from None
            except (DecisionClientError, ImportError):
                raise
            except Exception:
                # SDK errors can contain provider bodies, prompts, or credentials.
                raise ClientTransportError("Copilot SDK operation failed") from None
            finally:
                if session is not None:
                    # A second cancellation must not abandon runtime cleanup.
                    cleanup = asyncio.create_task(self._cleanup_session(session, abort=not completed))
                    cancelled = False
                    while not cleanup.done():
                        try:
                            await asyncio.shield(cleanup)
                        except asyncio.CancelledError:
                            cancelled = True
                    cleanup.result()
                    if cancelled:
                        raise asyncio.CancelledError()

    async def _cleanup_session(self, session, *, abort: bool) -> None:
        # Best effort and bounded; preserve the original response/error. Residual
        # session files are also removed when the whole client is closed.
        if abort:
            with suppress(Exception):
                await asyncio.wait_for(session.abort(), timeout=5)
        with suppress(Exception):
            await asyncio.wait_for(session.disconnect(), timeout=5)
        with suppress(Exception):
            await asyncio.wait_for(self._runtime.delete_session(session.session_id), timeout=5)

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._runtime is not None:
                    try:
                        await asyncio.wait_for(self._runtime.stop(), timeout=self.timeout_seconds)
                    except TimeoutError:
                        raise ClientTimeoutError("Copilot shutdown timed out") from None
                    except Exception:
                        raise ClientTransportError("Copilot shutdown failed") from None
            finally:
                self._runtime = None
                if self._directory is not None:
                    self._directory.cleanup()
                    self._directory = None