"""Explicit client registration with lazy imports of optional backends."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, TypeVar

from .base import BaseDecisionClient


ClientType = TypeVar("ClientType", bound=type[BaseDecisionClient])
_BUILTINS = {
    "typesafe": ("src.clients.typesafe", "TypeSafeClient"),
    "github_copilot": ("src.clients.github_copilot_client", "GitHubCopilotClient"),
    "vllm": ("src.clients.vllm_client", "VLLMClient"),
}
_REGISTERED: dict[str, type[BaseDecisionClient]] = {}


def _normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("client name must be a nonempty string")
    return name.strip().lower()


def register_client(name: str) -> Callable[[ClientType], ClientType]:
    """Register an adapter class as a decorator; never silently overwrite one."""
    key = _normalize_name(name)

    def register(client_class: ClientType) -> ClientType:
        if not isinstance(client_class, type) or not issubclass(
            client_class, BaseDecisionClient
        ):
            raise TypeError("client must be a BaseDecisionClient subclass")
        if key in _BUILTINS or key in _REGISTERED:
            raise ValueError(f"Client already registered: {key}")
        _REGISTERED[key] = client_class
        return client_class

    return register


def available_clients() -> tuple[str, ...]:
    """List known names without importing backends or checking credentials."""
    return tuple(sorted(set(_BUILTINS) | set(_REGISTERED)))


def create_client(name: str, **kwargs: Any) -> BaseDecisionClient:
    """Instantiate an adapter. Model IDs and thinking settings are kwargs.

    Backend dependencies are imported only when that backend is selected.
    """
    key = _normalize_name(name)
    if key in _REGISTERED:
        client_class = _REGISTERED[key]
    elif key in _BUILTINS:
        module_name, class_name = _BUILTINS[key]
        client_class = getattr(import_module(module_name), class_name)
    else:
        choices = ", ".join(available_clients())
        raise ValueError(f"Unknown client {key!r}; available clients: {choices}")
    return client_class(**kwargs)