#!/usr/bin/env python3
"""Read-only, finite monitoring of the two remaining paper experiments."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "output/paper-monitor-20260924-v1"
JOBS = {
    "deadline": {
        "worker": 2078692, "finalizer": 2095781,
        "run": "output/qwen9-deadline-control-20260924-v1",
        "final": "output/qwen9-deadline-finalization-20260924-v1/status.json",
        "complete": ["verified_completed_analysis", "verified_abort_analysis"],
    },
    "gpt54": {
        "worker": 1804254, "finalizer": 1804228,
        "run": "output/paper-gpt54-canonical-anchor-20260924-v3",
        "final": "paper/generated-gpt/completion.json",
        "complete": ["verified_and_rendered"],
    },
}
ACTIVE = {"registered", "waiting", "running", "waiting_predecessor", "admission"}
FINAL_ACTIVE = {"waiting_terminal", "analyzing"}
TERMINAL = {"completed", "completed_with_failures"}


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return {"start_ticks": fields[19], "state": fields[0]}


def alive(pid, expected):
    current = identity(pid)
    return current is not None and current["state"] not in ("Z", "X") and (
        current["start_ticks"] == expected["start_ticks"])


def read(path, optional=False):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        if optional:
            return None
        raise


def snapshot(registered):
    result, alerts = {}, []
    for name, spec in JOBS.items():
        run = REPO / spec["run"]
        state = read(run / "status.json")
        final = read(REPO / spec["final"], optional=True)
        done = final is not None and final.get("status") in spec["complete"]
        worker_live = alive(spec["worker"], registered[name]["worker"])
        finalizer_live = alive(spec["finalizer"], registered[name]["finalizer"])
        ledger = run / "attempts.jsonl"
        age = time.time() - ledger.stat().st_mtime if ledger.exists() else None
        result[name] = {
            "source_status": state["status"], "calls": state.get("calls"),
            "worker_alive": worker_live, "finalizer_alive": finalizer_live,
            "finalizer_status": final.get("status") if final else None,
            "analysis_finished": done, "last_attempt_age_seconds": age,
        }
        if state["status"] in ACTIVE:
            if not worker_live:
                alerts.append(f"{name}: worker disappeared before terminal status")
            if age is not None and age > 1200:
                alerts.append(f"{name}: no persisted attempt for more than 20 minutes")
        elif state["status"] not in TERMINAL:
            alerts.append(f"{name}: experiment ended with {state['status']}")
        if not done and not finalizer_live:
            alerts.append(f"{name}: finalizer disappeared without completion evidence")
        if final and not done and final.get("status") not in FINAL_ACTIVE:
            alerts.append(f"{name}: finalizer reports {final.get('status')}")
        if final and final.get("status") == "verified_abort_analysis":
            alerts.append(f"{name}: verification completed for an aborted experiment")
    return {
        "at_utc": datetime.now(timezone.utc).isoformat(), "jobs": result,
        "alerts": alerts, "all_analyses_finished": all(r["analysis_finished"] for r in result.values()),
        "model_calls": 0, "resource_actions": 0,
    }


def persist(value):
    pending = ROOT / "status.next.json"
    with pending.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(ROOT / "status.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    registered = {}
    for name, spec in JOBS.items():
        registered[name] = {}
        for role in ("worker", "finalizer"):
            found = identity(spec[role])
            if found is None or found["state"] in ("Z", "X"):
                raise RuntimeError(f"{name}: {role} is not live at monitor admission")
            registered[name][role] = found
    if args.check:
        print(json.dumps(snapshot(registered), indent=2))
        return 0
    ROOT.mkdir(exist_ok=False)
    with (ROOT / "monitor-start.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "identity": identity(os.getpid()),
                   "jobs": JOBS, "process_identities": registered,
                   "maximum_seconds": 86400, "poll_seconds": 60}, stream, indent=2)
    deadline = time.monotonic() + 86400
    with (ROOT / "events.jsonl").open("x") as events:
        while True:
            try:
                result = snapshot(registered)
            except (OSError, ValueError, KeyError) as exc:
                result = {
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                    "jobs": {}, "alerts": [f"Monitor read failed: {type(exc).__name__}: {exc}"],
                    "all_analyses_finished": False, "model_calls": 0, "resource_actions": 0,
                }
            result["monitor_status"] = (
                "attention_needed" if result["alerts"] else
                "completed" if result["all_analyses_finished"] else
                "deadline_exceeded" if time.monotonic() >= deadline else "monitoring")
            persist(result)
            events.write(json.dumps(result) + "\n")
            events.flush()
            os.fsync(events.fileno())
            if result["monitor_status"] != "monitoring":
                print(json.dumps(result), flush=True)
                return 0 if result["monitor_status"] == "completed" else 1
            time.sleep(min(60, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    raise SystemExit(main())
