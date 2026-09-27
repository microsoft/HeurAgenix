"""Run configured graph evaluation with python -m src.benchmark --config PATH."""

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path

from .config import load_config
from .runner import run_experiment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run topology-only graph evaluation with the configured model provider")
    parser.add_argument("--config", type=Path, required=True, help="YAML benchmark configuration (.yaml/.yml)")
    parser.add_argument("--output", type=Path, help="Override output directory (relative to current directory)")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress display on stderr")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.output:
        config = replace(config, run=replace(config.run, output_dir=args.output.resolve()))
    result = asyncio.run(run_experiment(config, progress=not args.no_progress))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()