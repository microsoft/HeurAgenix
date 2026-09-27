#!/usr/bin/env python3
"""Single-process, finite, durable offline wait and full canonical matrix export."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True
from scripts import export_full_canonical_matrix as exporter

suite = exporter.suite
ASTRA = REPO / "output/full-matrix-astra-20260925-v2"
ROOT = REPO / "output/full-matrix-offline-finalization-20260925-v1"
MAX_WAIT = 72 * 60 * 60
INTERVAL = 60
PRODUCERS = ("full_matrix_qwen27", "full_matrix_laya", "full_matrix_trained",
             "run_paper_gpt_anchor", "run_canonical_astra")
ABORTED = {"aborted", "interrupted", "failed", "failed_verification", "verification_failed"}


class DeadlineExpired(TimeoutError):
    pass


def deadline_alarm(_signum, _frame):
    raise DeadlineExpired("Registered 72-hour-or-shorter finalization deadline expired.")


def now():
    return datetime.now(timezone.utc).isoformat()


def local(path):
    path = Path(path).absolute()
    suite.require(path.resolve().is_relative_to(REPO) and not path.is_symlink(),
                  "Finalization paths must be unaliased repository-local paths.")
    return path


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(path, value, *, exclusive=False):
    destination = path if exclusive else path.with_name("." + path.name + ".next")
    with destination.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    if not exclusive:
        destination.replace(path)
    fsync_directory(path.parent)


class Journal:
    def __init__(self, root):
        self.root, self.lock = root, threading.Lock()

    def record(self, status, **values):
        with self.lock:
            row = {"status": status, "updated_at_utc": now(), "model_calls": 0, **values}
            with (self.root / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            write(self.root / "status.json", row)
            return row


def source_freeze():
    hashes = exporter.canonical.code_hashes()
    for path in (Path(__file__), Path(exporter.__file__), Path(exporter.matrix.__file__),
                 REPO / "tests/test_finalize_full_canonical_matrix.py",
                 REPO / "tests/test_export_full_canonical_matrix.py",
                 *(REPO / "scripts" / (name + ".py") for name in PRODUCERS)):
        hashes[str(path.relative_to(REPO))] = suite.digest(path)
    for lane in exporter.lanes(ASTRA):
        metadata = lane.root / ("manifest.json" if lane.kind in ("reference", "gpt54", "astra")
                                else "registration.json")
        signature = metadata.with_suffix(".sha256")
        suite.require(metadata.is_file() and signature.is_file(),
                      f"Register every fixed source before preparing the watcher: {metadata}")
        data = suite.read_json(metadata)
        for path in (metadata, signature):
            hashes[str(path.relative_to(REPO))] = suite.digest(path)
        suite.require(suite.digest(metadata) == suite.read_json(signature),
                      f"Source registration hash differs: {metadata}")
        for field in ("source_sha256", "code_sha256", "files_sha256"):
            for name, digest in data.get(field, {}).items():
                if Path(name).suffix == ".py":
                    suite.require(name not in hashes or hashes[name] == digest,
                                  f"Inconsistent registered code: {name}")
                    hashes[name] = digest
    verify_freeze(hashes)
    return hashes


def verify_freeze(hashes):
    for name, digest in hashes.items():
        path = local(REPO / name)
        suite.require(path.is_file() and suite.digest(path) == digest,
                      f"Frozen offline source or registration changed: {name}")


def prepare(root, max_wait_seconds=MAX_WAIT):
    root = local(root)
    suite.require(type(max_wait_seconds) is int and 0 < max_wait_seconds <= MAX_WAIT,
                  "Waiting limit must be positive and at most 72 hours.")
    suite.require(not root.exists(), "Existing finalization directory; no overwrite or resume.")
    hashes = source_freeze()
    registration = {
        "schema": 1, "created_at_utc": now(), "deadline_unix": time.time() + max_wait_seconds,
        "max_wait_seconds": max_wait_seconds, "interval_seconds": INTERVAL,
        "source_sha256": hashes, "astra_root": str(ASTRA),
        "profiles": [lane.name for lane in exporter.lanes(ASTRA)],
        "output": str(root / "export"), "model_calls": 0,
        "policy": "One claim, no resume/retry, all ten terminal verified lanes or no completed table.",
    }
    root.mkdir(parents=True, exist_ok=False)
    write(root / "registration.json", registration, exclusive=True)
    write(root / "registration.sha256", suite.digest(root / "registration.json"), exclusive=True)
    Journal(root).record("prepared", deadline_unix=registration["deadline_unix"])
    return registration


def load(root):
    suite.require(suite.digest(root / "registration.json") ==
                  suite.read_json(root / "registration.sha256"), "Watcher registration changed.")
    registration = suite.read_json(root / "registration.json")
    suite.require(registration["schema"] == 1 and registration["astra_root"] == str(ASTRA)
                  and registration["profiles"] == [lane.name for lane in exporter.lanes(ASTRA)]
                  and registration["output"] == str(root / "export")
                  and registration["interval_seconds"] == INTERVAL
                  and 0 < registration["max_wait_seconds"] <= MAX_WAIT,
                  "Watcher registration identity differs.")
    verify_freeze(registration["source_sha256"])
    return registration


def statuses():
    """Only lifecycle metadata. Never requests, attempts, scores, summaries, or proofs."""
    result = {}
    for lane in exporter.lanes(ASTRA):
        path = lane.terminal
        if not path.is_file():
            alternatives = (lane.run / "status.json", lane.root / "status.json")
            path = next((p for p in alternatives if p.is_file()), None)
        state = suite.read_json(path) if path else {}
        status = state.get("status", "not_started")
        terminal_present = lane.terminal.is_file()
        sealed = lane.seal.is_file()
        result[lane.name] = {
            "root": str(lane.root), "status": status, "terminal_present": terminal_present,
            "seal_present": sealed, "ready": terminal_present and sealed and status in exporter.COMPLETE,
            "blocked": status in ABORTED,
        }
    return result


def process_identity():
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": os.getpid(), "startticks": int(fields[19]), "started_at_utc": now()}


def durable_export(root):
    seal = suite.read_json(root / "export-seal.json")
    expected = {"decision_matrix-episodes.csv", "decision_matrix-task-size.csv",
                "strong_anchors-episodes.csv", "strong_anchors-task-size.csv",
                "full-matrix-tables.tex", "verified-summary.json"}
    suite.require(seal["status"] == "completed" and set(seal["artifact_sha256"]) == expected,
                  "Incomplete final export artifact inventory.")
    for name in sorted(expected | {"export-seal.json"}):
        path = local(root / name)
        suite.require(path.is_file() and (name == "export-seal.json" or
                      suite.digest(path) == seal["artifact_sha256"][name]),
                      f"Export artifact changed: {name}")
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    fsync_directory(root)


def run(root, *, clock=time.time, monotonic=time.monotonic, sleep=time.sleep,
        read_statuses=statuses, export=exporter.export):
    root = local(root)
    suite.require(root.is_dir(), "Prepare the exclusive watcher root first.")
    write(root / "runner.json", process_identity(), exclusive=True)
    journal = Journal(root)
    stop = threading.Event()
    heartbeat = None
    previous_alarm = None
    try:
        registration = load(root)
        remaining = min(registration["max_wait_seconds"], registration["deadline_unix"] - clock())
        deadline = monotonic() + max(0, remaining)
        while True:
            load(root)
            states = read_statuses()
            if any(value["blocked"] for value in states.values()):
                journal.record("blocked_terminal_source", lanes=states)
                return 2
            if monotonic() >= deadline or clock() >= registration["deadline_unix"]:
                journal.record("deadline_expired", lanes=states, deadline_unix=registration["deadline_unix"])
                return 2
            if all(value["ready"] for value in states.values()):
                break
            journal.record("waiting", lanes=states, deadline_unix=registration["deadline_unix"])
            sleep(min(INTERVAL, max(0, deadline - monotonic())))
        load(root)
        journal.record("verifying", lanes=states)

        def pulse():
            while not stop.wait(INTERVAL):
                journal.record("verifying", lanes=states, detail="Offline verification still running.")

        heartbeat = threading.Thread(target=pulse, name="offline-verification-heartbeat", daemon=True)
        heartbeat.start()
        previous_alarm = signal.signal(signal.SIGALRM, deadline_alarm)
        signal.setitimer(signal.ITIMER_REAL, max(.001, min(
            deadline - monotonic(), registration["deadline_unix"] - clock())))
        result = export(ASTRA, root / "export")
        load(root)
        durable_export(root / "export")
        if monotonic() >= deadline or clock() >= registration["deadline_unix"]:
            raise DeadlineExpired("Offline verification exceeded the registered deadline.")
        signal.setitimer(signal.ITIMER_REAL, 0)
        stop.set()
        heartbeat.join()
        failed_sources = [name for name, value in states.items()
                          if value["status"] == "completed_with_failures"]
        final = "completed_with_source_failures" if failed_sources else "completed"
        report = journal.record(final, result=result, source_failures=failed_sources)
        write(root / "completion.json", {
            **report, "registration_sha256": suite.digest(root / "registration.json"),
            "export_seal_sha256": suite.digest(root / "export/export-seal.json"),
        }, exclusive=True)
        return 1 if failed_sources else 0
    except DeadlineExpired as exc:
        stop.set()
        if heartbeat:
            heartbeat.join()
        journal.record("deadline_expired", error=str(exc))
        return 2
    except Exception as exc:
        stop.set()
        if heartbeat:
            heartbeat.join()
        journal.record("failed_verification", error=f"{type(exc).__name__}: {exc}"[:2000])
        return 2
    finally:
        stop.set()
        if previous_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "check"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--max-wait-seconds", type=int, default=MAX_WAIT)
    args = parser.parse_args()
    if args.command == "prepare":
        registration = prepare(args.root, args.max_wait_seconds)
        print(json.dumps({"status": "prepared", "root": str(args.root),
                          "deadline_unix": registration["deadline_unix"]}))
    elif args.command == "run":
        raise SystemExit(run(args.root))
    else:
        print(json.dumps(statuses(), indent=2))


if __name__ == "__main__":
    main()
