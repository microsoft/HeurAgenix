"""Strict safe YAML configuration; no secrets or arbitrary constructor kwargs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

import yaml
import httpx

from src.datasets import available_datasets


class _ConfigLoader(yaml.SafeLoader):
    """SafeLoader with string-only, unique mapping keys (no silent overrides).

    Merge keys are intentionally unsupported: configurations are explicit rather
    than inherited. Standard safe YAML scalar/list/mapping values are supported.
    """

    def construct_mapping(self, node, deep=False):
        if not isinstance(node, yaml.MappingNode):
            raise ValueError("YAML configuration requires mappings")
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("YAML mapping keys must be strings")
            if key in mapping:
                raise ValueError(f"Duplicate YAML key at line {key_node.start_mark.line + 1}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


@dataclass(frozen=True)
class RunConfig:
    seed: int
    repetitions: int
    max_calls: int
    output_dir: Path


@dataclass(frozen=True)
class DataConfig:
    datasets: tuple[str, ...]
    data_dir: Path


@dataclass(frozen=True)
class GraphExclusionConfig:
    """Pinned prior graph artifact defining a source-node-disjoint development pool."""

    path: Path
    sha256: str
    overlap: str = "source_nodes"

    def __post_init__(self) -> None:
        if (not isinstance(self.sha256, str) or len(self.sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.sha256)):
            raise ValueError("exclude_graphs.sha256 must be a lowercase SHA-256 digest")
        if self.overlap != "source_nodes":
            raise ValueError("Only source_nodes exclusion is supported")


@dataclass(frozen=True)
class PresentationConfig:
    """Shared model-visible task wording and lossless topology serialization."""

    prompt_variant: str = "baseline"
    graph_representation: str = "nodes_edges"

    def __post_init__(self) -> None:
        if self.prompt_variant not in ("baseline", "algorithmic_v1"):
            raise ValueError("Unsupported presentation.prompt_variant")
        if self.graph_representation not in ("nodes_edges", "adjacency_list_v1"):
            raise ValueError("Unsupported presentation.graph_representation")


@dataclass(frozen=True)
class SamplingConfig:
    node_counts: tuple[int, ...]
    samples_per_size: int
    max_attempts: int
    method: str = "connected_frontier"
    exclude_graphs: GraphExclusionConfig | None = None


@dataclass(frozen=True)
class TaskConfig:
    adjacency_questions: int
    distance_questions_per_threshold: int
    distance_thresholds: tuple[int, ...]


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    timeout_seconds: float
    think: bool | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    constrain_choices: bool | None = None
    seed: int | None = None
    output_format: str | None = None

    def __post_init__(self) -> None:
        fields = {
            "typesafe": set(),
            "github_copilot": {"think", "max_tokens", "reasoning_effort", "output_format"},
            "vllm": {"think", "max_tokens", "base_url", "temperature", "top_p", "top_k",
                     "presence_penalty", "constrain_choices", "seed", "output_format"},
        }
        if not isinstance(self.provider, str) or self.provider not in fields:
            raise ValueError("model.provider must be typesafe, github_copilot, or vllm")
        for name in {"think", "max_tokens", "reasoning_effort", "base_url", "temperature",
                     "top_p", "top_k", "presence_penalty", "constrain_choices", "seed", "output_format"}:
            if getattr(self, name) is not None and name not in fields[self.provider]:
                raise ValueError(f"model.{name} is not supported by this provider")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model.model must be nonempty")
        if (type(self.timeout_seconds) not in (int, float)
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be finite and positive")
        if self.think is not None and type(self.think) is not bool:
            raise ValueError("model.think must be bool or null")
        if self.output_format is not None and self.output_format not in ("json", "answer_only"):
            raise ValueError("model.output_format must be json or answer_only")
        if self.constrain_choices is not None and type(self.constrain_choices) is not bool:
            raise ValueError("model.constrain_choices must be bool or null")
        if self.constrain_choices is True and (
                self.think is not False or self.output_format != "answer_only"):
            raise ValueError("model.constrain_choices requires think=false and output_format=answer_only")
        if self.max_tokens is not None:
            _integer(self.max_tokens, "model.max_tokens")
        if self.seed is not None:
            _integer(self.seed, "model.seed", 0)
        if self.top_k is not None and (
                type(self.top_k) is not int or self.top_k != -1 and self.top_k < 1):
            raise ValueError("model.top_k must be -1 or a positive integer")
        if self.presence_penalty is not None and (
                type(self.presence_penalty) not in (int, float)
                or not math.isfinite(self.presence_penalty) or not -2 <= self.presence_penalty <= 2):
            raise ValueError("model.presence_penalty must be finite and in [-2, 2]")
        if self.reasoning_effort is not None:
            if self.reasoning_effort not in ("low", "medium", "high", "xhigh", "max"):
                raise ValueError("Unsupported model.reasoning_effort")
            if self.think is False:
                raise ValueError("model.reasoning_effort conflicts with think=false")
        for name, value, upper in (("temperature", self.temperature, 2), ("top_p", self.top_p, 1)):
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)
                                      or not 0 <= value <= upper or (name == "top_p" and value == 0)):
                raise ValueError(f"Invalid model.{name}")
        if self.base_url is not None:
            try:
                if not isinstance(self.base_url, str):
                    raise ValueError()
                urlsplit(self.base_url)  # Reject malformed bracketed IPv6 hosts.
                url = httpx.URL(self.base_url)
                if (url.scheme not in ("http", "https") or not url.host or url.username
                        or url.password or url.query or url.fragment):
                    raise ValueError()
            except (ValueError, httpx.InvalidURL):
                raise ValueError("model.base_url must be HTTP(S) without credentials/query/fragment") from None

    def client_kwargs(self) -> dict:
        """Allowlisted effective settings; credentials never enter configuration."""
        kwargs = {"model": self.model, "timeout_seconds": self.timeout_seconds}
        if self.provider != "typesafe":
            kwargs.update(think=self.think, max_tokens=self.max_tokens if self.max_tokens is not None else 4096)
            if self.output_format is not None:
                kwargs["output_format"] = self.output_format
        if self.provider == "github_copilot":
            kwargs["reasoning_effort"] = self.reasoning_effort
        elif self.provider == "vllm":
            kwargs.update(base_url=self.base_url if self.base_url is not None else "http://127.0.0.1:8000/v1",
                          temperature=self.temperature if self.temperature is not None else 0.0,
                          top_p=self.top_p if self.top_p is not None else 1.0, seed=self.seed)
            for name in ("top_k", "presence_penalty", "constrain_choices"):
                if getattr(self, name) is not None:
                    kwargs[name] = getattr(self, name)
        return kwargs


@dataclass(frozen=True)
class BenchmarkConfig:
    schema_version: int
    run: RunConfig
    data: DataConfig
    sampling: SamplingConfig
    tasks: TaskConfig
    model: ModelConfig
    presentation: PresentationConfig = PresentationConfig()

    @property
    def planned_graphs(self) -> int:
        return len(self.data.datasets) * len(self.sampling.node_counts) * self.sampling.samples_per_size

    @property
    def planned_questions(self) -> int:
        per_graph = self.tasks.adjacency_questions + (
            self.tasks.distance_questions_per_threshold * len(self.tasks.distance_thresholds)
        )
        return self.planned_graphs * per_graph

    def snapshot(self) -> dict:
        snapshot = json.loads(json.dumps(asdict(self), default=str))
        # Keep original Jev snapshots stable; null optional fields need no keys.
        snapshot["model"] = {key: value for key, value in snapshot["model"].items() if value is not None}
        if self.presentation == PresentationConfig():
            snapshot.pop("presentation")
        if self.sampling.exclude_graphs is None:
            snapshot["sampling"].pop("exclude_graphs")
        return snapshot

    @property
    def sha256(self) -> str:
        body = json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode()).hexdigest()


def _keys(value: object, expected: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name}: expected exactly these keys: {', '.join(sorted(expected))}")
    return value


def _integer(value: object, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _integers(value: object, name: str, minimum: int, *, empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list) or (not value and not empty):
        raise ValueError(f"{name} must be a nonempty array")
    result = tuple(_integer(v, name, minimum) for v in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _path(value: object, parent: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("paths must be nonempty strings")
    path = Path(value).expanduser()
    return (parent / path).resolve()


def load_config(path: str | Path) -> BenchmarkConfig:
    path = Path(path).resolve()
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Benchmark config must be a .yaml or .yml file")
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.load(stream, Loader=_ConfigLoader)
    except yaml.YAMLError as exc:
        # PyYAML's usual exception text echoes input. Avoid logging config values.
        mark = getattr(exc, "problem_mark", None)
        location = f" at line {mark.line + 1}" if mark is not None else ""
        raise ValueError(f"Invalid or unsupported YAML configuration{location}") from None
    required = {"schema_version", "run", "data", "sampling", "tasks", "model"}
    if not isinstance(data, dict) or not required <= set(data) or set(data) - required - {"presentation"}:
        raise ValueError("config: missing required keys or unsupported sections")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Only schema_version = 1 is supported")
    run = _keys(data["run"], {"seed", "repetitions", "max_calls", "output_dir"}, "run")
    source = _keys(data["data"], {"datasets", "data_dir"}, "data")
    sampling = data["sampling"]
    required_sampling = {"method", "node_counts", "samples_per_size", "max_attempts"}
    if (not isinstance(sampling, dict) or not required_sampling <= set(sampling)
            or set(sampling) - required_sampling - {"exclude_graphs"}):
        raise ValueError("sampling: missing required keys or unsupported settings")
    exclusion = None
    if "exclude_graphs" in sampling:
        raw = _keys(sampling["exclude_graphs"], {"path", "sha256", "overlap"}, "exclude_graphs")
        exclusion = GraphExclusionConfig(_path(raw["path"], path.parent), raw["sha256"], raw["overlap"])
    presentation = PresentationConfig()
    if "presentation" in data:
        presentation = PresentationConfig(**_keys(
            data["presentation"], {"prompt_variant", "graph_representation"}, "presentation"))
    tasks = _keys(data["tasks"], {"adjacency_questions", "distance_questions_per_threshold", "distance_thresholds"}, "tasks")
    model = data["model"]
    required_model_keys = {"provider", "model", "timeout_seconds"}
    optional_model_keys = {"think", "max_tokens", "reasoning_effort", "base_url", "temperature",
                           "top_p", "top_k", "presence_penalty", "constrain_choices", "seed", "output_format"}
    if (not isinstance(model, dict) or not required_model_keys <= set(model)
            or set(model) - required_model_keys - optional_model_keys):
        raise ValueError("model: missing required keys or unsupported settings; credentials are not allowed")
    names = source["datasets"]
    if not isinstance(names, list) or not names or any(
        not isinstance(n, str) or n not in available_datasets() for n in names
    ):
        raise ValueError("data.datasets must contain registered dataset IDs")
    if len(names) != len(set(names)):
        raise ValueError("data.datasets must not contain duplicates")
    if sampling["method"] != "connected_frontier":
        raise ValueError("Only connected_frontier sampling is supported")
    adjacency = _integer(tasks["adjacency_questions"], "adjacency_questions", 0)
    distance = _integer(tasks["distance_questions_per_threshold"], "distance_questions_per_threshold", 0)
    thresholds = _integers(tasks["distance_thresholds"], "distance_thresholds", 2, empty=True)
    if adjacency % 2 or distance % 2:
        raise ValueError("Question counts must be even for equal yes/no quotas")
    if bool(distance) != bool(thresholds) or not (adjacency or distance):
        raise ValueError("Enable at least one task; distance count and thresholds must both be enabled or disabled")
    model_config = ModelConfig(**model)
    config = BenchmarkConfig(
        schema_version=1,
        run=RunConfig(_integer(run["seed"], "seed", 0),
                      _integer(run["repetitions"], "repetitions"),
                      _integer(run["max_calls"], "max_calls"),
                      _path(run["output_dir"], path.parent)),
        data=DataConfig(tuple(names), _path(source["data_dir"], path.parent)),
        sampling=SamplingConfig(_integers(sampling["node_counts"], "node_counts", 3),
                                _integer(sampling["samples_per_size"], "samples_per_size"),
                                _integer(sampling["max_attempts"], "max_attempts"), exclude_graphs=exclusion),
        tasks=TaskConfig(adjacency, distance, thresholds),
        model=model_config,
        presentation=presentation,
    )
    if config.planned_questions * config.run.repetitions > config.run.max_calls:
        raise ValueError("Planned calls exceed run.max_calls (checked before sampling or network access)")
    return config