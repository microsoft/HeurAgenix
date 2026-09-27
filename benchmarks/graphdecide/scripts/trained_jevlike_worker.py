#!/usr/bin/env python3
"""Isolated inference worker. Stdin carries only public DecisionRequests."""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
from pathlib import Path
import sys
import signal
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", choices=("decider", "kev"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    supervisor = int(os.environ["TRAINED_JEVLIKE_SUPERVISOR_PID"])
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL) != 0 or os.getppid() != supervisor:
        raise RuntimeError("Cannot bind worker lifetime to its supervisor")
    root = args.root.resolve()
    sys.path.insert(0, str(root / "deps"))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4",
                      MKL_NUM_THREADS="4", TORCH_FORCE_WEIGHTS_ONLY_LOAD="1")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif os.environ.get("TRAINED_JEVLIKE_GPU_HANDOFF") != "approved":
        raise RuntimeError("GPU worker requires explicit supervisor handoff")
    from scripts import trained_jevlike_adapter as adapter
    wire = sys.stdout

    def emit(value):
        wire.write(json.dumps(value, allow_nan=False) + "\n")
        wire.flush()

    with contextlib.redirect_stdout(sys.stderr):
        try:
            start = time.perf_counter()
            model, metadata = adapter.initialize(root, args.candidate, args.device)
            emit({"status": "ready", "runtime": metadata,
                  "loading_seconds": time.perf_counter() - start})
        except Exception as exc:
            emit({"status": "fatal", "error_type": type(exc).__name__, "error": str(exc)[:1200]})
            return 1
        for line in sys.stdin:
            try:
                request = adapter.request_from_dict(json.loads(line))
                start = time.perf_counter()
                result = adapter.infer(model, args.candidate, request)
                emit({"status": "success", "result": result,
                      "worker_seconds": time.perf_counter() - start})
            except Exception as exc:
                from src.clients.base import UnsupportedRequestError
                nonfatal = isinstance(exc, UnsupportedRequestError) or type(exc).__name__ == "ContextOverflow"
                emit({"status": "failure", "fatal": not nonfatal,
                      "error_type": type(exc).__name__, "error": str(exc)[:1200]})
                if not nonfatal:
                    return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
